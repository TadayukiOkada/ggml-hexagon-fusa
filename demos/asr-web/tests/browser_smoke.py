import asyncio,json,argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--url",default="http://127.0.0.1:8765")
parser.add_argument("--browser",required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path=args.browser,headless=True,args=['--no-sandbox','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'])
  context=await browser.new_context(permissions=['microphone'],viewport={'width':1280,'height':900})
  page=await context.new_page();errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  await page.add_init_script("window.testTracks=[];const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async(...args)=>{const stream=await original(...args);window.testTracks.push(...stream.getTracks());return stream;}")
  await page.goto(args.url);await page.wait_for_function("() => document.querySelector('#status').textContent.includes('準備完了')")
  for engine in ['both','nemo','sensevoice']:
   await page.select_option('#engine',engine);await page.click('#start');await page.wait_for_function("() => document.querySelector('#status').textContent.includes('収録中')")
   await page.wait_for_timeout(650);await page.click('#stop');await page.wait_for_function("() => document.querySelector('#status').textContent.startsWith('完了')")
   assert await page.locator('#save-audio').is_enabled()
   assert await page.evaluate("() => window.testTracks.every(t=>t.readyState==='ended')")
   for e in (['nemo','sensevoice'] if engine=='both' else [engine]):assert 'MOCK' in await page.locator('#'+e+' .final').inner_text()
  await page.select_option('#engine','both');await page.select_option('#scenario','late');await page.click('#start');await page.wait_for_function("() => document.querySelector('#status').textContent.includes('収録中')");await page.wait_for_timeout(300);await page.click('#stop');await page.click('#cancel');await page.wait_for_timeout(2200)
  assert '取り消しました' in await page.locator('#status').inner_text()
  await page.select_option('#scenario','busy');await page.click('#start');await page.wait_for_function("() => document.querySelector('#status').textContent.includes('収録中')");await page.wait_for_timeout(300);await page.click('#stop');await page.wait_for_function("() => document.querySelector('#status').textContent.startsWith('完了')")
  # Exercise tags, WAV input, export, and the exact 30-second sample boundary.
  import io,wave
  wav=io.BytesIO()
  with wave.open(wav,'wb') as f:
   f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(b'\0'*1002)
  await page.select_option('#scenario','events')
  await page.set_input_files('#file',{'name':'tail.wav','mimeType':'audio/wav','buffer':wav.getvalue()})
  await page.wait_for_function("() => document.querySelector('#status').textContent.startsWith('完了')")
  await page.wait_for_function("() => document.querySelector('#sensevoice .analysis').textContent.includes('Laughter')")
  async with page.expect_download() as download:
   await page.click('#save')
  path=await(await download.value).path();data=json.loads(Path(path).read_text())
  done=[e for e in data['events'] if e['type']=='engine.done']
  assert len(done)==2 and all(e['samples']==501 for e in done) and done[0]['pcm_sha256']==done[1]['pcm_sha256']
  for scenario in ['unknown','no-tags','disconnect','malformed','replacement']:
   await page.select_option('#scenario',scenario)
   await page.click('#start');await page.wait_for_function("() => document.querySelector('#status').textContent.includes('収録中')")
   await page.wait_for_timeout(1100);await page.click('#stop');await page.wait_for_function("() => document.querySelector('#status').textContent.startsWith('完了')")
  await page.select_option('#scenario','normal')
  await page.click('#start');await page.wait_for_function("() => document.querySelector('#status').textContent.includes('収録中')")
  await page.wait_for_function("() => document.querySelector('#status').textContent.startsWith('完了')",timeout=40000)
  async with page.expect_download() as download:
   await page.click('#save')
  path=await(await download.value).path();data=json.loads(Path(path).read_text())
  assert data['events'][-1]['samples']==480000
  assert all(e['samples']==480000 for e in data['events'] if e['type']=='engine.done')
  await page.screenshot(path=str(args.output/'mock-desktop.png'),full_page=True)
  await page.set_viewport_size({'width':390,'height':844});await page.screenshot(path=str(args.output/'mock-mobile.png'),full_page=True)
  # A separate context refuses the microphone; controls must recover.
  denied=await browser.new_context(permissions=[])
  denied_page=await denied.new_page()
  await denied_page.add_init_script("navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Microphone denied','NotAllowedError')}")
  await denied_page.goto(args.url);await denied_page.click('#start')
  await denied_page.wait_for_function("() => document.querySelector('#status').textContent.includes('Microphone denied')")
  assert await denied_page.locator('#start').is_enabled()
  await denied.close()
  assert not errors,errors
  print(json.dumps({'browser':browser.version,'errors':errors,'tests':['fake mic three engines','stop flush','cancel during stop','busy isolation','501-sample WAV and PCM identity','unknown/absent/multiple tags','30-second automatic stop','microphone tracks released','microphone refusal injection','desktop/mobile screenshot']},ensure_ascii=False))
  await browser.close()
asyncio.run(main())
