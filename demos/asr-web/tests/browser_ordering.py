"""Browser-only protocol injection: ordering and stale recording isolation."""
import argparse
import asyncio
import json
from playwright.async_api import async_playwright

async def main(args):
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=args.browser,headless=True,
            args=['--no-sandbox','--use-fake-device-for-media-stream'])
        context=await browser.new_context(permissions=['microphone'])
        page=await context.new_page();sockets=[];starts=[];errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        def route(ws):
            sockets.append(ws)
            def message(raw):
                if not isinstance(raw,str):return
                event=json.loads(raw)
                if event['type']=='start':
                    starts.append(event)
                    ws.send(json.dumps(dict(type='session.ready',session_id='s'+str(len(starts)),
                        generation=event['generation'],is_mock=True,engines=['nemo'])))
            ws.on_message(message)
        await page.route_web_socket('**/api/session',route)
        await page.goto(args.url)
        await page.wait_for_function("() => !document.querySelector('#start').disabled")
        await page.select_option('#engine','nemo')
        await page.click('#start')
        await page.wait_for_function("() => document.querySelector('#status').textContent.includes('収録中')")
        first=dict(session_id='s1',generation=starts[0]['generation'],engine='nemo',segment_id='nemo-1',is_mock=True)
        sockets[0].send(json.dumps(dict(type='result.partial',revision=2,final=False,text='new hypothesis',**first)))
        sockets[0].send(json.dumps(dict(type='result.partial',revision=1,final=False,text='out of order',**first)))
        await page.wait_for_function("() => document.querySelector('#nemo .partial').textContent==='new hypothesis'")
        sockets[0].send(json.dumps(dict(type='result.final',revision=3,final=True,text='final text',**first)))
        sockets[0].send(json.dumps(dict(type='result.partial',revision=4,final=False,text='late partial',**first)))
        await page.wait_for_function("() => document.querySelector('#nemo .final').textContent==='final text'")
        assert await page.locator('#nemo .partial').inner_text()==''
        await page.click('#cancel')
        await page.wait_for_function("() => !document.querySelector('#start').disabled")
        await page.click('#start')
        await page.wait_for_function("() => document.querySelector('#status').textContent.includes('収録中')")
        sockets[1].send(json.dumps(dict(type='result.final',revision=9,final=True,text='old recording',**first)))
        second=dict(first,session_id='s2',generation=starts[1]['generation'])
        sockets[1].send(json.dumps(dict(type='result.partial',revision=1,final=False,text='current recording',**second)))
        await page.wait_for_function("() => document.querySelector('#nemo .partial').textContent==='current recording'")
        sockets[1].send(json.dumps(dict(type='result.final',revision=2,final=True,text='',**second)))
        await page.wait_for_function("() => document.querySelector('#nemo .partial').textContent===''")
        assert await page.locator('#nemo .final').inner_text()==''
        assert await page.locator('#nemo .final').inner_text()==''
        await page.click('#cancel')
        assert not errors,errors
        print(json.dumps(dict(browser=browser.version,tests=['revision ordering','partial after final','previous recording generation'],errors=errors)))
        await browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--browser',required=True)
    asyncio.run(main(parser.parse_args()))
