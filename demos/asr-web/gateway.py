"""Local ASR comparison gateway. Mock transport; no audio is persisted."""
import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
import time
import uuid
import wave

from aiohttp import web, WSMsgType, ClientSession, ClientTimeout
from native import NativeEngine, load_profiles

STATIC = Path(__file__).parent / 'static'
ENGINES = ('nemo', 'sensevoice')
SCENARIOS = ('normal', 'slow', 'busy', 'disconnect', 'malformed', 'replacement', 'late', 'events', 'unknown', 'no-tags')
MAX_SAMPLES = 480000


def selection(data):
    engines = data.get('engines', list(ENGINES))
    if not isinstance(engines, list) or not engines or len(set(engines)) != len(engines) or any(e not in ENGINES for e in engines):
        raise ValueError('invalid engines')
    if data.get('sample_rate', 16000) != 16000:
        raise ValueError('sample_rate must be 16000')
    if data.get('scenario', 'normal') not in SCENARIOS:
        raise ValueError('invalid scenario')
    return engines


class Recording:
    def __init__(self, data, send, profiles=None):
        self.engines = selection(data)
        self.profile = data.get('profile', 'mock')
        self.is_mock = self.profile == 'mock'
        self.native = {}
        if not self.is_mock:
            if self.profile not in (profiles or {}):
                raise ValueError('profile is not configured')
            if any(e not in profiles[self.profile] for e in self.engines):
                raise ValueError('selected engine is not configured')
            self.native = {e: NativeEngine(e, profiles[self.profile][e], self) for e in self.engines}
        self.scenario = data.get('scenario', 'normal')
        self.generation = data.get('generation')
        if not isinstance(self.generation, int):
            raise ValueError('integer generation required')
        self.id = uuid.uuid4().hex
        self.send = send
        self.samples = 0
        self.hashes = {e: hashlib.sha256() for e in self.engines}
        self.counts = {e: 0 for e in self.engines}
        self.failed = set()
        self.tasks = []
        self.stopping = False
        self.cancelled = False
        self.revision = 0
        self.started = time.monotonic()

    async def emit(self, kind, **fields):
        if not self.cancelled:
            await self.send(dict(type=kind, session_id=self.id, recording_id=self.id,
                                 generation=self.generation, is_mock=self.is_mock, **fields))

    async def error(self, engine, code, message=None):
        self.failed.add(engine)
        await self.emit('engine.error', engine=engine, code=code, message=message, retryable=True)

    async def start(self):
        results=await asyncio.gather(*(a.start() for a in self.native.values()), return_exceptions=True)
        failures=[e for e,r in zip(self.native,results) if isinstance(r,Exception)]
        if self.native and len(failures)==len(self.native):
            raise ValueError('選択したエンジンの準備ができていません')
        await self.emit('session.ready', engines=self.engines, max_samples=MAX_SAMPLES)
        for engine in failures: await self.native[engine].fail('native_not_ready')
        if not self.is_mock: return
        if self.scenario == 'busy' and 'sensevoice' in self.engines:
            await self.error('sensevoice', 'busy')

    async def push(self, pcm):
        if self.stopping or len(pcm) % 2 or len(pcm) > 3200 or self.samples + len(pcm)//2 > MAX_SAMPLES:
            raise ValueError('PCM frame invalid, too large, or recording limit exceeded')
        self.samples += len(pcm)//2
        if not self.is_mock:
            for engine, adapter in self.native.items():
                if engine not in self.failed:
                    try: adapter.push(pcm)
                    except OverflowError: await adapter.fail('queue_overflow')
            await self.emit('audio.ack', samples=self.samples)
            return
        for engine in self.engines:
            if engine not in self.failed:
                self.hashes[engine].update(pcm)
                self.counts[engine] += len(pcm)//2
        if self.scenario in ('disconnect', 'malformed') and self.samples >= 1600 and 'sensevoice' in self.engines and 'sensevoice' not in self.failed:
            await self.error('sensevoice', self.scenario)
        if 'nemo' in self.engines and self.samples // 8000 > self.revision:
            self.revision = self.samples // 8000
            text = ['これはモックの途中結果です。', 'これはモックです。'][self.revision % 2] if self.scenario == 'replacement' else 'モック音声を受信しています。'
            await self.result('nemo', False, text)
        await self.emit('audio.ack', samples=self.samples)

    async def result(self, engine, final, text):
        fields = dict(engine=engine, segment_id=engine+'-1', revision=self.revision+int(final),
                      final=final, text=text, audio_start_ms=0, audio_end_ms=self.samples/16)
        if engine == 'sensevoice':
            fields['analysis'] = dict(language='ja', emotion='NEUTRAL', events=['Speech'],
                                     itn='withitn', raw_tags=['<|ja|>', '<|NEUTRAL|>', '<|Speech|>', '<|withitn|>'], confidence=None)
        if engine == 'sensevoice':
            if self.scenario == 'events':
                fields['analysis'].update(emotion='HAPPY', events=['Laughter','Applause'], raw_tags=['<|ja|>','<|HAPPY|>','<|Laughter|>','<|Applause|>','<|withitn|>'])
            if self.scenario in ('unknown','no-tags'):
                fields['analysis'].update(language=None, emotion=None, events=[], itn=None, raw_tags=['<|unrecognized_tag|>'] if self.scenario=='unknown' else [])
        await self.emit('result.final' if final else 'result.partial', **fields)

    async def finish_engine(self, engine):
        if not self.is_mock:
            await self.native[engine].finish()
            return
        if engine in self.failed:
            return
        delay = 2 if self.scenario in ('slow', 'late') and engine == 'sensevoice' else .08
        await self.emit('engine.processing', engine=engine)
        await asyncio.sleep(delay)
        await self.result(engine, True, '【MOCK】これは入力内容を認識した結果ではありません。' if self.samples else '')
        await self.emit('engine.done', engine=engine, samples=self.counts[engine], pcm_sha256=self.hashes[engine].hexdigest(), elapsed_ms=round((time.monotonic()-self.started)*1000))

    async def stop(self):
        self.stopping = True
        await asyncio.gather(*(self.finish_engine(e) for e in self.engines))
        await self.emit('session.done', samples=self.samples, failed_engines=sorted(self.failed))

    async def cancel(self, notify=True):
        if notify:
            await self.emit('session.cancelled')
        self.cancelled = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.gather(*(a.close() for a in self.native.values()))


def origin_allowed(request):
    return request.headers.get('Origin') == f'{request.scheme}://{request.host}'


async def websocket(request):
    if not origin_allowed(request):
        raise web.HTTPForbidden(text='same-origin required')
    ws = web.WebSocketResponse(max_msg_size=4096, heartbeat=20)
    await ws.prepare(request)
    recording = None
    async def send(event):
        await ws.send_json(event)
    try:
        async for msg in ws:
            try:
                if msg.type == WSMsgType.BINARY:
                    if recording is None:
                        raise ValueError('start required')
                    await recording.push(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if not isinstance(data, dict):
                        raise ValueError('JSON object required')
                    kind = data.get('type')
                    if kind == 'start':
                        if recording is not None:
                            raise ValueError('one recording per WebSocket')
                        recording = Recording(data, send, request.app[PROFILES])
                        await recording.start()
                    elif recording and data.get('session_id') == recording.id and data.get('generation') == recording.generation:
                        if kind == 'stop' and not recording.stopping:
                            recording.stopping = True
                            recording.tasks.append(asyncio.create_task(recording.stop()))
                        elif kind == 'cancel':
                            await recording.cancel()
                        else:
                            raise ValueError('invalid control or state')
                    else:
                        raise ValueError('invalid session or generation')
                elif msg.type == WSMsgType.ERROR:
                    break
            except (ValueError, TypeError, KeyError) as exc:
                await ws.send_json(dict(type='session.error', code='invalid_request', message=str(exc), is_mock=recording.is_mock if recording else True))
                if recording:
                    await recording.cancel()
                await ws.close(code=1008)
                break
    finally:
        if recording:
            await recording.cancel(notify=False)
    return ws


async def transcribe(request):
    if not origin_allowed(request):
        raise web.HTTPForbidden(text='same-origin required')
    try:
        data = await request.read()
        with wave.open(io.BytesIO(data), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, 'NONE') or wav.getnframes() > MAX_SAMPLES:
                raise ValueError('16kHz mono PCM16 WAV, maximum 30 seconds required')
            pcm = wav.readframes(wav.getnframes())
            if len(pcm) != wav.getnframes()*2:
                raise ValueError('truncated WAV')
        events = []
        async def send(event):
            events.append(event)
        recording = Recording(dict(generation=0, profile=request.query.get('profile','mock'), engines=request.query.get('engines', 'nemo,sensevoice').split(',')), send, request.app[PROFILES])
        try:
            await recording.start()
            for pos in range(0, len(pcm), 3200):
                await recording.push(pcm[pos:pos+3200])
                if not recording.is_mock: await asyncio.sleep(.1)
            await recording.stop()
            return web.json_response(dict(is_mock=recording.is_mock, events=events))
        finally:
            await recording.cancel(notify=False)
    except (ValueError, wave.Error, EOFError) as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc


async def capabilities(request):
    result = dict(profile='mock', is_mock=True, profiles=[dict(id='mock', ready=True), dict(id='evk', ready=False), dict(id='sa8255p', ready=False)],
        engines={e: dict(ready=True, backend='mock', device=None, model='scripted', build_id=None, partial=e=='nemo', tags=e=='sensevoice') for e in ENGINES},
        limits=dict(sample_rate=16000, max_seconds=30, max_frame_bytes=3200, pending_seconds=2), scenarios=SCENARIOS)
    profiles=[dict(id="mock", ready=True, is_mock=True, engines=result['engines'])]
    async with ClientSession(timeout=ClientTimeout(total=2)) as client:
        async def probe(profile, engine, url):
            try:
                async with client.get(url+'/ready') as response:
                    info=await response.json()
                    return engine, dict(ready=response.status==200 and info.get('ready') is True,
                                        backend=info.get('backend',profile), device=info.get('device'),
                                        model=info.get('model'), build_id=info.get('build_id'),
                                        partial=engine=='nemo', tags=engine=='sensevoice')
            except Exception: return engine, dict(ready=False)
        for profile in ('cpu','evk','sa8255p'):
            engines=dict(await asyncio.gather(*(probe(profile,e,u) for e,u in request.app[PROFILES].get(profile,{}).items())))
            profiles.append(dict(id=profile, is_mock=False, configured=profile in request.app[PROFILES], ready=any(e['ready'] for e in engines.values()), engines=engines))
    result['profiles']=profiles
    return web.json_response(result)


@web.middleware
async def headers(request, handler):
    # Host validation also prevents DNS rebinding against the local gateway.
    if request.url.host not in ('localhost', '127.0.0.1', '::1'):
        raise web.HTTPForbidden(text='loopback Host required')
    response = await handler(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'"
    return response


PROFILES = web.AppKey("profiles", dict)

def create_app(profiles=None):
    app = web.Application(client_max_size=MAX_SAMPLES*2+4096, middlewares=[headers])
    app[PROFILES] = profiles or {}
    app.router.add_get('/api/capabilities', capabilities)
    app.router.add_get('/api/session', websocket)
    app.router.add_post('/api/transcribe', transcribe)
    async def index(request):
        return web.FileResponse(STATIC/'index.html')
    app.router.add_get('/', index)
    app.router.add_static('/static/', STATIC)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    web.run_app(create_app(load_profiles(args.config)), host='127.0.0.1', port=args.port)
