import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
from aiohttp import web
from aiohttp.test_utils import TestServer
sys.path.insert(0,str(Path(__file__).parents[1]))
from native import NativeEngine

class Sink:
    def __init__(self):self.events=[]
    async def emit(self,kind,**fields):self.events.append(dict(type=kind,**fields))
    async def error(self,engine,code,message=None):self.events.append(dict(type='engine.error',engine=engine,code=code))

class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_unresponsive_websocket_handshake_times_out(self):
        disconnected=asyncio.Event()
        async def silent_server(reader,writer):
            try:
                await reader.readuntil(b'\r\n\r\n')
                await reader.read()
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()
        server=await asyncio.start_server(silent_server,'127.0.0.1',0)
        port=server.sockets[0].getsockname()[1]
        engine=NativeEngine('sensevoice',f'http://127.0.0.1:{port}',Sink())
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(engine.start(),7)
            await engine.close()
            await asyncio.wait_for(disconnected.wait(),1)
            self.assertTrue(engine.client.closed)
        finally:
            await engine.close()
            server.close();await server.wait_closed()

    async def test_nemo_empty_final_replaces_partial(self):
        async def handler(request):
            ws=web.WebSocketResponse();await ws.prepare(request)
            await ws.send_json(dict(type='session.created'))
            async for msg in ws:
                if msg.type!=web.WSMsgType.TEXT:continue
                import json
                if json.loads(msg.data)['type']=='session.update':
                    await ws.send_json(dict(type='session.updated'))
                    continue
                for kind,text,revision in [('delta','tentative',1),('completed','',2)]:
                    await ws.send_json(dict(type='conversation.item.input_audio_transcription.'+kind,
                                            transcript=text,segment_id=1,revision=revision))
                await ws.send_json(dict(type='input_audio_buffer.committed'));break
            return ws
        app=web.Application();app.router.add_get('/v1/audio/transcriptions/realtime',handler)
        async with TestServer(app) as server:
            sink=Sink();engine=NativeEngine('nemo',str(server.make_url('/')).rstrip('/'),sink)
            try:
                await engine.start();engine.push(bytes(640));await engine.finish()
                results=[e for e in sink.events if e['type'].startswith('result.')]
                self.assertEqual([(e['type'],e['text']) for e in results],
                                 [('result.partial','tentative'),('result.final','')])
                self.assertFalse(engine.failed)
                self.assertEqual(sink.events[-1]['type'],'engine.done')
            finally:await engine.close()

    async def test_queue_bound(self):
        engine=NativeEngine('sensevoice','http://127.0.0.1',Sink())
        for _ in range(20):engine.push(b'\0'*3200)
        with self.assertRaises(OverflowError):engine.push(b'\0\0')
        self.assertEqual(engine.pending,64000)

    async def test_normal_and_invalid_native(self):
        for malformed in (False,True):
            async def handler(request):
                ws=web.WebSocketResponse();await ws.prepare(request)
                await ws.send_json(dict(type='session.ready'))
                async for msg in ws:
                    if msg.type==web.WSMsgType.TEXT:
                        await ws.send_json(dict(type='result.final',text='',segment_id=1,revision=1,analysis={},samples=320) if not malformed else dict(type='result.final',delta='ambiguous'))
                        await ws.send_json(dict(type='session.done'));break
                return ws
            app=web.Application();app.router.add_get('/api/session',handler)
            async with TestServer(app) as server:
                sink=Sink();engine=NativeEngine('sensevoice',str(server.make_url('/')).rstrip('/'),sink)
                try:
                    await engine.start();engine.push(b'\0'*640);await engine.finish()
                    self.assertEqual(engine.failed,malformed)
                    self.assertEqual(len([e for e in sink.events if e['type']=='result.final']),int(not malformed))
                    self.assertEqual(len([e for e in sink.events if e['type']=='engine.done']),int(not malformed))
                finally:await engine.close()

if __name__=='__main__':unittest.main()
