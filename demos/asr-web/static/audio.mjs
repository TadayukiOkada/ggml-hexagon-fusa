// Windowed-sinc low-pass resampler. Absolute sample positions survive blocks.
export class PCMResampler {
  constructor(rate, emit) {
    this.rate=rate; this.emit=emit; this.step=rate/16000; this.radius=32;
    this.samples=[]; this.base=0; this.total=0; this.output=0; this.chunk=[];
  }
  push(channels) {
    if (!channels.length) return;
    for(let i=0;i<channels[0].length;i++) {
      let x=0; for(const c of channels) x+=c[i];
      this.samples.push(x/channels.length); this.total++;
    }
    this.drain(false);
  }
  drain(final) {
    const cutoff=.94*Math.min(1,16000/this.rate);
    const target=Math.floor(this.total*16000/this.rate);
    while(final ? this.output<target : this.output*this.step+this.radius<this.total) {
      const p=this.output*this.step, mid=Math.floor(p); let sum=0, norm=0;
      for(let j=mid-this.radius+1;j<=mid+this.radius;j++) {
        const d=j-p, z=Math.PI*d*cutoff;
        const w=(Math.abs(z)<1e-12?1:Math.sin(z)/z)*(.5+.5*Math.cos(Math.PI*d/this.radius));
        const value=j<0||j>=this.total?0:this.samples[j-this.base];
        sum+=value*w; norm+=w;
      }
      const x=Math.max(-1,Math.min(1,sum/norm));
      this.chunk.push(Math.round(x*(x<0?32768:32767))); this.output++;
      if(this.chunk.length===320) this.flushChunk();
    }
    const keep=Math.max(0,Math.floor(this.output*this.step)-this.radius);
    if(keep>this.base) {this.samples.splice(0,keep-this.base);this.base=keep;}
  }
  flushChunk() {
    if(!this.chunk.length) return;
    const bytes=new ArrayBuffer(this.chunk.length*2), view=new DataView(bytes);
    this.chunk.forEach((x,i)=>view.setInt16(i*2,x,true)); this.chunk=[];this.emit(bytes);
  }
  finish(){this.drain(true);this.flushChunk();}
}

export function readWav(buffer) {
  const v=new DataView(buffer), str=(p,n)=>String.fromCharCode(...new Uint8Array(buffer,p,n));
  if(buffer.byteLength<44||str(0,4)!=='RIFF'||str(8,4)!=='WAVE') throw Error('WAVファイルを選んでください。');
  let fmt=null, pcm=null;
  for(let p=12;p+8<=buffer.byteLength;) {
    const n=v.getUint32(p+4,true), start=p+8;
    if(start+n>buffer.byteLength) throw Error('WAVが途中で切れています。');
    if(str(p,4)==='fmt '&&n>=16) fmt=[v.getUint16(start,true),v.getUint16(start+2,true),v.getUint32(start+4,true),v.getUint16(start+14,true)];
    if(str(p,4)==='data') pcm=buffer.slice(start,start+n);
    p=start+n+(n%2);
  }
  if(!fmt||fmt.join(',')!=='1,1,16000,16'||!pcm||pcm.byteLength%2||pcm.byteLength>960000) throw Error('16kHz / mono / PCM16、30秒以内のWAVが必要です。');
  return pcm;
}
