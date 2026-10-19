"""Bounded, independent native transport per engine. No subprocess inference."""
import asyncio
import hashlib
import json
from urllib.parse import urlsplit
from aiohttp import ClientSession, ClientTimeout, WSMsgType


class NativeEngine:
    def __init__(self, engine, url, recording):
        self.engine=engine; self.url=url; self.recording=recording
        self.queue=asyncio.Queue(); self.pending=0; self.samples=0
        self.digest=hashlib.sha256(); self.ws=None; self.client=None
        self.sender=None; self.receiver=None; self.finished=asyncio.Event()
        self.failed=False; self.revisions={}; self.server_samples=None

    async def start(self):
        self.client=ClientSession(timeout=ClientTimeout(total=None, sock_connect=5))
        # No heartbeat: the native servers infer inside the socket handler and cannot answer
        # pings meanwhile. The queue limit and finish() bound the waits instead.
        self.ws=await asyncio.wait_for(self.client.ws_connect(self.url+('/v1/audio/transcriptions/realtime' if self.engine=='nemo' else '/api/session'),max_msg_size=1024*1024),5)
        event=await asyncio.wait_for(self.ws.receive_json(),5)
        expected='session.created' if self.engine=='nemo' else 'session.ready'
        if event.get('type')!=expected:raise ValueError('native engine is busy or not ready')
        if self.engine=='nemo':
            await self.ws.send_json(dict(type='session.update',session=dict(sample_rate=16000,language='ja-JP',automatic_punctuation=True)))
            updated=await asyncio.wait_for(self.ws.receive_json(),5)
            if updated.get('type')!='session.updated':raise ValueError('NeMo configuration rejected')
        self.sender=asyncio.create_task(self.send_loop())
        self.receiver=asyncio.create_task(self.receive_loop())

    def push(self,pcm):
        if self.failed:return
        if self.pending+len(pcm)>64000:raise OverflowError('engine queue exceeds 2 seconds')
        self.pending+=len(pcm);self.queue.put_nowait(pcm)

    async def fail(self,code,message=None):
        if not self.failed:
            self.failed=True
            await self.recording.error(self.engine,code,message=message)
        self.finished.set()

    async def send_loop(self):
        try:
            while True:
                pcm=await self.queue.get()
                if pcm is None:
                    await self.ws.send_json({'type':'input_audio_buffer.commit' if self.engine=='nemo' else 'stop'})
                    return
                await self.ws.send_bytes(pcm)
                self.pending-=len(pcm);self.samples+=len(pcm)//2;self.digest.update(pcm)
        except asyncio.CancelledError:raise
        except Exception:await self.fail('native_send_failed')

    async def receive_loop(self):
        try:
            async for message in self.ws:
                if message.type!=WSMsgType.TEXT:raise ValueError('native disconnected')
                event=json.loads(message.data);kind=event.get('type','')
                if kind=='error':raise ValueError(event.get('message') or event.get('error') or event.get('code') or 'native inference failed')
                if kind in ('input_audio_buffer.committed','session.done'):
                    self.finished.set();return
                if kind.startswith('conversation.item.input_audio_transcription.') or kind=='result.final':
                    final=kind.endswith('.completed') or kind=='result.final'
                    segment=event.get('segment_id');revision=event.get('revision');text=event.get('transcript') if self.engine=='nemo' else event.get('text')
                    if segment is None or not isinstance(revision,int) or not isinstance(text,str):raise ValueError('native full-hypothesis fields missing')
                    if revision<=self.revisions.get(str(segment),-1):continue
                    self.revisions[str(segment)]=revision
                    fields=dict(engine=self.engine,segment_id=f'{self.engine}-{segment}',revision=revision,final=final,text=text,
                                audio_start_ms=0,audio_end_ms=event.get('audio_processed',self.samples/16000)*1000)
                    if self.engine=='sensevoice':
                        fields['analysis']=event['analysis'];fields['ids']=event.get('ids');fields['compute_ms']=event.get('compute_ms');self.server_samples=event.get('samples')
                    await self.recording.emit('result.final' if final else 'result.partial',**fields)
            raise ValueError(f'native disconnected before completion (close={self.ws.close_code})')
        except asyncio.CancelledError:raise
        except Exception as exc:await self.fail('native_invalid_or_disconnected',str(exc))

    async def finish(self):
        if self.failed:return
        await self.recording.emit('engine.processing',engine=self.engine)
        self.queue.put_nowait(None)
        try:await asyncio.wait_for(self.finished.wait(),60)
        except asyncio.TimeoutError:await self.fail('native_timeout')
        if not self.failed:
            if self.server_samples is not None and self.server_samples!=self.samples:
                await self.fail('native_sample_mismatch');return
            await self.recording.emit('engine.done',engine=self.engine,samples=self.samples,pcm_sha256=self.digest.hexdigest())

    async def close(self):
        for task in (self.sender,self.receiver):
            if task:task.cancel()
        await asyncio.gather(*(t for t in (self.sender,self.receiver) if t),return_exceptions=True)
        if self.ws:await self.ws.close()
        if self.client:await self.client.close()


def load_profiles(path):
    if not path:return {}
    data=json.loads(path.read_text())
    for name,engines in data.items():
        if name not in ('cpu','evk','sa8255p') or not isinstance(engines,dict):raise ValueError('invalid profile')
        for engine,url in engines.items():
            parsed=urlsplit(url)
            if engine not in ('nemo','sensevoice') or parsed.scheme!='http' or parsed.hostname not in ('localhost','127.0.0.1','::1') or parsed.path not in ('','/') or parsed.query or parsed.fragment or parsed.username:
                raise ValueError('native endpoints must be loopback HTTP origins; use local tunnels')
            engines[engine]=url.rstrip('/')
    return data
