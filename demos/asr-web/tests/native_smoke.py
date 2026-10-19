"""Optional live integration: use running gateway/native servers and an explicit WAV."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import wave
from aiohttp import ClientSession

async def main(args):
    with wave.open(str(args.wav)) as wav:
        if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,16000):
            raise ValueError('16kHz mono PCM16 required')
        pcm=wav.readframes(wav.getnframes())
    if len(pcm)>960000:raise ValueError('maximum 30 seconds')
    engines=args.engines.split(',')
    async with ClientSession() as client:
        async with client.ws_connect(args.url+'/api/session',headers={'Origin':args.url}) as ws:
            await ws.send_json(dict(type='start',generation=1,profile=args.profile,engines=engines))
            ready=await ws.receive_json();events=[ready]
            assert ready['type']=='session.ready',ready
            for pos in range(0,len(pcm),3200):
                await ws.send_bytes(pcm[pos:pos+3200]);await asyncio.sleep(.1)
            await ws.send_json(dict(type='stop',session_id=ready['session_id'],generation=1))
            while True:
                event=await asyncio.wait_for(ws.receive_json(),65);events.append(event)
                if event['type']=='session.done':break
    args.output.write_text(json.dumps(events,ensure_ascii=False,indent=2))
    assert not [e for e in events if e['type']=='engine.error'],events
    done=[e for e in events if e['type']=='engine.done']
    assert len(done)==len(engines)
    assert all(e['samples']==len(pcm)//2 and e['pcm_sha256']==hashlib.sha256(pcm).hexdigest() for e in done)
    print('Native delivery verified:',len(pcm)//2,'samples; PCM SHA-256',hashlib.sha256(pcm).hexdigest())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--profile',choices=['cpu','evk','sa8255p'],required=True)
    parser.add_argument('--engines',default='nemo,sensevoice')
    parser.add_argument('--wav',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
