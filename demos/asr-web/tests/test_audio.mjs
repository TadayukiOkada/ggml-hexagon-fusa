import test from 'node:test';
import assert from 'node:assert/strict';
import {PCMResampler,readWav} from '../static/audio.mjs';
function resample(rate,signal,block=128){const out=[];const r=new PCMResampler(rate,b=>out.push(Buffer.from(b)));for(let p=0;p<signal.length;p+=block)r.push([signal.slice(p,p+block)]);r.finish();return Buffer.concat(out);}
for(const rate of [44100,48000]){
  test(`${rate}: sample count, block invariance and tail`,()=>{
    const signal=Float32Array.from({length:rate+173},(_,i)=>.5*Math.sin(2*Math.PI*1000*i/rate));
    const a=resample(rate,signal),b=resample(rate,signal,997);
    assert.equal(a.length,Math.floor(signal.length*16000/rate)*2);assert.deepEqual(a,b);
  });
  test(`${rate}: rejects aliasing`,()=>{
    const amp=f=>{const out=resample(rate,Float32Array.from({length:rate},(_,i)=>.5*Math.sin(2*Math.PI*f*i/rate)));let sum=0;for(let p=200;p<out.length-200;p+=2)sum+=(out.readInt16LE(p)/32768)**2;return Math.sqrt(sum/(out.length/2-200));};
    assert.ok(amp(1000)>.3);assert.ok(amp(12000)<.005);
  });
}
test('WAV rejects malformed or wrong format',()=>assert.throws(()=>readWav(new ArrayBuffer(44))));
