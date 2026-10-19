import asyncio
import hashlib
import importlib.util
import io
from pathlib import Path
import unittest
import wave
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from aiohttp.test_utils import TestClient, TestServer

spec=importlib.util.spec_from_file_location('gateway', Path(__file__).parents[1]/'gateway.py')
gateway=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)

class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client=TestClient(TestServer(gateway.create_app()))
        await self.client.start_server()
        self.origin=str(self.client.make_url('/')).rstrip('/')

    async def asyncTearDown(self):
        await self.client.close()

    async def connect(self, engines=None, scenario='normal'):
        ws=await self.client.ws_connect('/api/session', headers={'Origin':self.origin})
        await ws.send_json(dict(type='start', generation=7, engines=engines or ['nemo','sensevoice'], scenario=scenario))
        ready=await ws.receive_json()
        self.assertEqual(ready['type'],'session.ready')
        return ws, dict(session_id=ready['session_id'],generation=7)

    async def finish(self, ws, control):
        await ws.send_json(dict(type='stop',**control))
        events=[]
        while True:
            event=await asyncio.wait_for(ws.receive_json(),4)
            events.append(event)
            if event['type']=='session.done':return events

    async def test_selection_and_identical_tail(self):
        pcm=bytes(range(256))*19+b'\x12\x34'
        for engines in (['nemo'],['sensevoice'],['nemo','sensevoice']):
            ws,control=await self.connect(engines)
            for pos in range(0,len(pcm),640):await ws.send_bytes(pcm[pos:pos+640])
            events=await self.finish(ws,control)
            done=[e for e in events if e['type']=='engine.done']
            self.assertEqual({e['engine'] for e in done},set(engines))
            for e in done:
                self.assertEqual(e['samples'],len(pcm)//2)
                self.assertEqual(e['pcm_sha256'],hashlib.sha256(pcm).hexdigest())
            await ws.close()

    async def test_failure_isolated(self):
        for scenario in ('busy','disconnect','malformed'):
            ws,control=await self.connect(scenario=scenario)
            await ws.send_bytes(b'\0'*3200)
            events=await self.finish(ws,control)
            self.assertEqual(events[-1]['failed_engines'],['sensevoice'])
            self.assertEqual([e['engine'] for e in events if e['type']=='engine.done'],['nemo'])
            await ws.close()

    async def test_cancel_during_stop(self):
        ws,control=await self.connect(scenario='late')
        await ws.send_bytes(b'\0'*640)
        await ws.send_json(dict(type='stop',**control))
        await ws.send_json(dict(type='cancel',**control))
        while (await ws.receive_json())['type']!='session.cancelled':pass
        with self.assertRaises(asyncio.TimeoutError):await asyncio.wait_for(ws.receive_json(),.2)
        await ws.close()

    async def test_invalid_generation(self):
        ws,control=await self.connect()
        await ws.send_json(dict(type='stop',session_id=control['session_id'],generation=6))
        self.assertEqual((await ws.receive_json())['type'],'session.error')
        await ws.close()

    async def test_reject_odd_frame(self):
        ws,control=await self.connect()
        await ws.send_bytes(b'x')
        self.assertEqual((await ws.receive_json())['type'],'session.error')
        await ws.close()

    async def test_unready_native_never_emits_ready(self):
        events=[]
        async def send(event):events.append(event)
        rec=gateway.Recording(dict(generation=1,profile='evk',engines=['sensevoice']),send,
                              {'evk':{'sensevoice':'http://127.0.0.1:1'}})
        try:
            with self.assertRaises(ValueError):await rec.start()
            self.assertFalse(any(e['type']=='session.ready' for e in events))
        finally:await rec.cancel(notify=False)

    async def test_limit(self):
        events=[]
        async def send(e):events.append(e)
        rec=gateway.Recording(dict(generation=1),send)
        for _ in range(300):await rec.push(b'\0'*3200)
        with self.assertRaises(ValueError):await rec.push(b'\0\0')
        self.assertEqual(rec.samples,480000)

    async def test_origin_and_static(self):
        resp=await self.client.get('/')
        self.assertEqual(resp.status,200)
        self.assertIn('Content-Security-Policy',resp.headers)
        with self.assertRaises(Exception):
            await self.client.ws_connect('/api/session',headers={'Origin':'http://evil.invalid'})
        resp=await self.client.get('/api/capabilities')
        self.assertTrue((await resp.json())['is_mock'])

    async def test_wav(self):
        buf=io.BytesIO()
        with wave.open(buf,'wb') as f:
            f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(b'\0'*1002)
        resp=await self.client.post('/api/transcribe',data=buf.getvalue(),headers={'Origin':self.origin})
        self.assertEqual(resp.status,200)
        data=await resp.json()
        self.assertEqual(data['events'][-1]['samples'],501)
        resp=await self.client.post('/api/transcribe',data=b'bad',headers={'Origin':self.origin})
        self.assertEqual(resp.status,400)

if __name__=='__main__':unittest.main()
