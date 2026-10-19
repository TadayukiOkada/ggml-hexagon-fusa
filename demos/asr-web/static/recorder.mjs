import {PCMResampler} from './audio.mjs';
class Recorder extends AudioWorkletProcessor {
  constructor(){
    super(); this.active=true;this.count=0;
    this.resampler=new PCMResampler(sampleRate, bytes=>{
      const remaining=480000-this.count, n=Math.min(remaining,bytes.byteLength/2);
      if(n>0){const out=bytes.slice(0,n*2);this.count+=n;this.port.postMessage({type:'pcm',bytes:out},[out]);}
      if(this.count>=480000&&this.active){this.active=false;this.port.postMessage({type:'limit'});}
    });
    this.port.onmessage=event=>{if(event.data==='flush'){this.active=false;this.resampler.finish();this.port.postMessage({type:'flushed'});}};
  }
  process(inputs){if(this.active&&inputs[0]?.length)this.resampler.push(inputs[0]);return true;}
}
registerProcessor('asr-recorder',Recorder);
