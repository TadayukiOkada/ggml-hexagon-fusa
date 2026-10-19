import {readWav} from './audio.mjs';
const $=id=>document.getElementById(id);
let generation=0, current=null, events=[], audio=[], settings=null, state='idle', lastMock=true;
const status=text=>{$('status').textContent=text;};
function controls(next){
  state=next;const idle=next==='idle';
  for(const id of ['start','engine','device','preset','scenario','file','profile'])$(id).disabled=!idle;
  $('stop').disabled=next!=='recording';$('cancel').disabled=idle;
  $('save-audio').disabled=!idle||!audio.length;
}
function clearResults(){for(const e of ['nemo','sensevoice'])for(const cls of ['partial','final','analysis','metrics'])$(e).querySelector('.'+cls).textContent='';}
function engineStatus(e,text){$(e).querySelector('.engine-status').textContent=text;}
function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function receive(r,event){
  if(current!==r||event.generation!==r.generation||event.session_id!==r.id&&event.type!=='session.ready')return;
  lastMock=event.is_mock;events.push({...event,browser_elapsed_ms:performance.now()-r.started});
  const mode=event.is_mock?' · MOCK':'';
  if(event.type==='session.ready'){r.id=event.session_id;$('recording').textContent='収録ID '+r.id.slice(0,12);r.resolve();return;}
  if(event.type==='audio.ack'){r.acked=event.samples;return;}
  if(event.engine){
    if(!['nemo','sensevoice'].includes(event.engine)){fail(r,'不正なエンジン応答');return;}
    const card=$(event.engine);
    if(event.type.startsWith('result.')){
      const key=event.engine+':'+event.segment_id;if(key in r.finals&&!event.final)return;const old=r.revisions[key]??-1;if(!Number.isInteger(event.revision)||event.revision<=old)return;
      r.revisions[key]=event.revision;
      if(event.final){r.finals[key]=event.text;card.querySelector('.final').textContent=Object.entries(r.finals).filter(([k])=>k.startsWith(event.engine+':')).map(([,v])=>v).filter(Boolean).join('\n');}else card.querySelector('.partial').textContent=event.text;
      if(event.final)card.querySelector('.partial').textContent='';
      engineStatus(event.engine,(event.final?'確定':'途中結果')+mode);
      if(event.analysis){const labels=[['言語',event.analysis.language],['感情',event.analysis.emotion],['イベント',event.analysis.events.join(', ')],['ITN',event.analysis.itn],['raw',event.analysis.raw_tags.join(' ')]];
        card.querySelector('.analysis').replaceChildren(...labels.map(([k,v])=>{const span=document.createElement('span');span.className='tag';span.textContent=k+': '+(v===null||v===''?'未返却':v);return span;}));}
    }
    if(event.type==='engine.processing')engineStatus(event.engine,'解析中');
    if(event.type==='engine.error'){engineStatus(event.engine,'エラー: '+event.code);card.querySelector('.metrics').textContent=event.message||'';}
    if(event.type==='engine.done')card.querySelector('.metrics').textContent=`${event.samples} samples · ${(event.samples/16000).toFixed(2)}秒 · ${event.elapsed_ms??Math.round(performance.now()-r.started)} ms\nPCM SHA-256 ${event.pcm_sha256}`;
  }
  if(event.type==='session.done'){r.done=true;status(event.failed_engines.length?'完了（一部エンジンにエラー）':'完了'+mode);controls('idle');current=null;r.ws.close();}
}
async function openSession(){
  controls('starting');clearResults();events=[];audio=[];settings=null;lastMock=$('profile').value==='mock';
  const r={generation:++generation,id:null,sent:0,acked:0,revisions:{},finals:{},started:performance.now()};current=r;
  const ready=new Promise((resolve,reject)=>{r.resolve=resolve;r.reject=reject;});
  const ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/session`);r.ws=ws;
  ws.onopen=()=>ws.send(JSON.stringify({type:'start',generation:r.generation,profile:$('profile').value,engines:$('engine').value==='both'?['nemo','sensevoice']:[$('engine').value],sample_rate:16000,scenario:$('scenario').value}));
  ws.onmessage=message=>{try{const e=JSON.parse(message.data);if(e.type==='session.error')throw Error(e.message);receive(r,e);}catch(error){fail(r,error.message);}};
  ws.onerror=()=>fail(r,'gatewayへの接続に失敗しました');
  ws.onclose=()=>{if(current===r&&!r.done)fail(r,'接続が切断されました');};
  r.timer=setTimeout(()=>fail(r,'接続がタイムアウトしました'),5000);
  try{await ready;}finally{clearTimeout(r.timer);}
  if(current!==r)throw Error('取り消しました');
  for(const e of ['nemo','sensevoice'])engineStatus(e,$('engine').value==='both'||$('engine').value===e?'収集中':'選択なし');
  return r;
}
function sendPCM(r,bytes){
  if(current!==r||r.ws.readyState!==WebSocket.OPEN)return;
  if(r.ws.bufferedAmount+bytes.byteLength>64000||r.sent-r.acked+bytes.byteLength/2>32000){fail(r,'送信待ちが2秒を超えました');return;}
  if(r.sent+bytes.byteLength/2>480000){fail(r,'収録上限を超えました');return;}
  r.ws.send(bytes);audio.push(bytes);r.sent+=bytes.byteLength/2;$('clock').textContent=`${(r.sent/16000).toFixed(2)} / 30秒`;
}
async function release(r,flush){
  if(r.releasing)return r.releasing;
  r.releasing=(async()=>{
    try{
      if(r.node&&flush){await new Promise((resolve,reject)=>{r.flushed=resolve;const timer=setTimeout(()=>reject(Error('録音末尾のflushがタイムアウトしました')),3000);r.flushTimer=timer;r.node.port.postMessage('flush');}).finally(()=>clearTimeout(r.flushTimer));}
    }finally{
      r.stream?.getTracks().forEach(t=>t.stop());r.source?.disconnect();r.node?.disconnect();
      if(r.context&&r.context.state!=='closed')await r.context.close();$('level').value=0;
    }
  })();return r.releasing;
}
async function fail(r,message){
  if(current!==r)return;r.reject?.(Error(message));current=null;r.ws?.close();
  await release(r,false).catch(()=>{});controls('idle');status(message);
}
$('start').onclick=async()=>{
  let r;
  try{
    r=await openSession();status('マイクの許可を待っています');
    const raw=$('preset').value==='raw';
    const stream=await navigator.mediaDevices.getUserMedia({audio:{deviceId:$('device').value?{exact:$('device').value}:undefined,echoCancellation:!raw,noiseSuppression:!raw,autoGainControl:!raw},video:false});
    if(current!==r){stream.getTracks().forEach(t=>t.stop());return;}
    r.stream=stream;settings=stream.getAudioTracks()[0].getSettings();$('details').textContent=JSON.stringify(settings,null,2);
    r.context=new AudioContext();await r.context.resume();await r.context.audioWorklet.addModule('/static/recorder.mjs');
    if(current!==r){await release(r,false);return;}
    r.node=new AudioWorkletNode(r.context,'asr-recorder');r.source=r.context.createMediaStreamSource(stream);
    r.node.port.onmessage=event=>{
      if(event.data.type==='pcm'){const bytes=event.data.bytes;sendPCM(r,bytes);const v=new DataView(bytes);let power=0;for(let i=0;i<v.byteLength;i+=2)power+=(v.getInt16(i,true)/32768)**2;$('level').value=Math.sqrt(power/(v.byteLength/2));}
      if(event.data.type==='flushed')r.flushed?.();
      if(event.data.type==='limit'&&state==='recording')stop();
    };
    r.source.connect(r.node);r.node.connect(r.context.destination);controls('recording');status('収録中 · 最大30秒');await devices();
  }catch(error){if(r)await fail(r,error.message);else{controls('idle');status(error.message);}}
};
async function stop(){const r=current;if(!r||state!=='recording')return;controls('stopping');status('末尾を送信し、確定結果を待っています');try{await release(r,true);if(current===r)r.ws.send(JSON.stringify({type:'stop',session_id:r.id,generation:r.generation}));}catch(error){await fail(r,error.message);}}
$('stop').onclick=stop;
$('cancel').onclick=async()=>{const r=current;if(!r)return;current=null;generation++;r.reject?.(Error('取り消しました'));if(r.ws.readyState===1&&r.id)r.ws.send(JSON.stringify({type:'cancel',session_id:r.id,generation:r.generation}));r.ws.close();await release(r,false).catch(()=>{});controls('idle');status('取り消しました');};
$('file').onchange=async event=>{
  let r;try{
    const file=event.target.files[0];if(!file)return;if(file.size>964096)throw Error('WAVは30秒以内にしてください。');
    const pcm=readWav(await file.arrayBuffer());r=await openSession();controls('recording');status('WAVを送信しています');
    for(let p=0;p<pcm.byteLength&&current===r;p+=3200){sendPCM(r,pcm.slice(p,p+3200));await new Promise(resolve=>setTimeout(resolve,100));}
    if(current===r)await stop();
  }catch(error){if(r)await fail(r,error.message);else status(error.message);}finally{event.target.value='';}
};
$('clear').onclick=()=>{clearResults();events=[];if(current)current.finals={};};
$('save').onclick=()=>download(new Blob([JSON.stringify({is_mock:lastMock,settings,events},null,2)],{type:'application/json'}),'asr-results.json');
$('save-audio').onclick=()=>{
  const n=audio.reduce((x,b)=>x+b.byteLength,0),head=new ArrayBuffer(44),v=new DataView(head),word=(p,s)=>[...s].forEach((c,i)=>v.setUint8(p+i,c.charCodeAt(0)));
  word(0,'RIFF');v.setUint32(4,36+n,true);word(8,'WAVE');word(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,16000,true);v.setUint32(28,32000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);word(36,'data');v.setUint32(40,n,true);
  download(new Blob([head,...audio],{type:'audio/wav'}),'asr-recording.wav');
};
async function devices(){if(!navigator.mediaDevices)return;const old=$('device').value;const list=await navigator.mediaDevices.enumerateDevices();$('device').replaceChildren(new Option('システムの既定',''),...list.filter(d=>d.kind==='audioinput').map((d,i)=>new Option(d.label||`マイク ${i+1}`,d.deviceId)));$('device').value=old;}
window.addEventListener('pagehide',()=>{if(current){current.stream?.getTracks().forEach(t=>t.stop());current.ws.close();}});
controls('starting');
try{const cap=await(await fetch('/api/capabilities')).json();for(const p of cap.profiles)if(p.id!=='mock')$('profile').add(new Option(p.id==='cpu'?'PC CPU（実推論）':p.id.toUpperCase(),p.id));for(const o of $('profile').options)if(o.value!=='mock')o.disabled=!cap.profiles.find(p=>p.id===o.value)?.ready;await devices();controls('idle');status('準備完了 · PCモック');}catch(error){status(error.message);$('start').disabled=true;}

$('profile').onchange=()=>{const mock=$('profile').value==='mock';document.querySelector('.badge').textContent=mock?'PC MOCK':$('profile').value.toUpperCase();document.querySelector('.notice').textContent=mock?'モックモード · 表示する文章・感情・イベントは試験用の固定データです。入力音声の認識は行いません。':'実推論モード · ローカル設定で指定したネイティブサーバへ音声を送ります。';status(mock?'準備完了 · PCモック':'実推論接続先を選択しました');};
