// Copyright (c) 2025 FunASR. MIT; derived from the pinned SenseVoice runtime.
// Shared frontend, SAN-M graph and greedy CTC; FuSa scheduler/flash paths preserved.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "funasr_backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const float LN_EPS = 1e-5f;

// Frontend accepts 16kHz mono float samples; file decoding/VAD stay in the CLI.
#include <utility>
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80,LFR_M=7,LFR_N=6;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;
static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
static void fftc(std::vector<float>&re,std::vector<float>&im,int n){
  for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
    for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
      float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
      re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}
}
static std::vector<float> compute_fbank(std::vector<float> wav,int&T_out){
  for(auto&v:wav)v*=32768.0f; std::vector<float> win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>> fb(NMEL,std::vector<float>(NBIN,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
    for(int k=0;k<NBIN;k++){float mf=melf(bw*k); if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  if(wav.size()<WINLEN){T_out=0;return {};}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1; std::vector<std::vector<float>> feat(T,std::vector<float>(NMEL));
  std::vector<float> re(NFFT),im(NFFT),fr(WINLEN); const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT; double mn=0; for(int i=0;i<WINLEN;i++)mn+=s[i]; mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn; for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1]; fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;} fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]); feat[t][m]=logf(e>fl?e:fl);}}
  const int pad=(LFR_M-1)/2; int Tl=(T+LFR_N-1)/LFR_N; std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
  for(int i=0;i<pad;i++)pd.push_back(feat[0]); for(int t=0;t<T;t++)pd.push_back(feat[t]);
  while((int)pd.size()<(Tl-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
  int D=LFR_M*NMEL; std::vector<float> out((size_t)Tl*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<LFR_M;j++)memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
  T_out=Tl; return out;
}

struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,vocab=25055,blank=0; };
struct model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
  ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;} };

static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T,bool flash){
  const int D=m.c.d_model,H=m.c.n_head,K=m.c.kernel;
  ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
  ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
  const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
  ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
  for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
    auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,sl,wj));}  // vp rows are contiguous: sl needs no copy
  ggml_tensor*o=funasr::attention(c,q,k,v,H,flash);  // flash attention on the HTP
  return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T,bool res,bool flash){
  auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  auto sa=sanm_attn(c,m,p+"self_attn.",h,T,flash); x=res?ggml_add(c,r,sa):sa; r=x;
  h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
  h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
  double inc=log(10000.0)/(depth/2.0-1.0);
  for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
    x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}

// SenseVoice detok: sentencepiece pieces (no byte-fallback in this vocab) -> join,
// "▁"(U+2581)->space; meta tokens <|lang|>/<|emo|>/<|event|>/<|itn|> dropped unless --keep-tags.
static std::string sv_trim(const std::string&s){size_t a=s.find_first_not_of(' ');if(a==std::string::npos)return "";size_t b=s.find_last_not_of(' ');return s.substr(a,b-a+1);}
static std::string detok_sv(const std::vector<int>&ids,const std::vector<std::string>&vocab,bool keep_tags){
  std::string s; for(int id:ids){ if(id<0||id>=(int)vocab.size())continue; const std::string&p=vocab[id];
    if(!keep_tags && p.size()>=2 && p[0]=='<' && p[1]=='|') continue;   // skip <|...|> meta
    s+=p; }
  const std::string lb="\xe2\x96\x81"; size_t pp; while((pp=s.find(lb))!=std::string::npos)s.replace(pp,3," ");
  return sv_trim(s);
}


#include "sensevoice_runtime.h"
namespace funasr {
struct sensevoice_runtime::impl {
  backend runtime;
  model m;
  std::vector<int> qtok;
  std::vector<std::string> vocab;
  impl(const std::string &path, const backend_options &options): runtime(options) {
  // load model
  gguf_init_params gp={false,&m.ctx_w}; gguf_context*gg=gguf_init_from_file(path.c_str(),gp);
  if(!gg){fprintf(stderr,"load gguf failed\n");throw std::runtime_error("load gguf failed");}
  auto rd=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
  m.c.d_model=rd("sv.output_size",512); m.c.n_head=rd("sv.attention_heads",4);
  m.c.num_blocks=rd("sv.num_blocks",50); m.c.tp_blocks=rd("sv.tp_blocks",20);
  m.c.kernel=rd("sv.kernel_size",11); m.c.vocab=rd("sv.vocab_size",25055); m.c.blank=rd("sv.blank_id",0);
  int qi=gguf_find_key(gg,"sv.query_tokens"); int nq=qi<0?0:(int)gguf_get_arr_n(gg,qi);
  qtok.resize(nq); for(int i=0;i<nq;i++) qtok[i]=((const int32_t*)gguf_get_arr_data(gg,qi))[i];
   {int ki=gguf_find_key(gg,"sv.vocab"); if(ki>=0){int nv=gguf_get_arr_n(gg,ki); vocab.resize(nv); for(int i=0;i<nv;i++){const char*s=gguf_get_arr_str(gg,ki,i); vocab[i]=s?s:"";}}}
  for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
  gguf_free(gg);
  runtime.prepare_weights(m.ctx_w,m.t);

  }
  recognition features(const std::vector<float> &fb,int T) {
    const int F=560,D=m.c.d_model,V=m.c.vocab,nq=qtok.size();
    const auto *emb=m.g("embed.weight");
    if(T<1 || fb.size()!=(size_t)T*F)throw std::runtime_error("invalid or empty fbank");
    int N=nq+T; std::vector<float> inp((size_t)N*F);
    for(int i=0;i<nq;i++) funasr::embedding_row(emb,qtok[i],F,&inp[(size_t)i*F]);
    memcpy(&inp[(size_t)nq*F], fb.data(), (size_t)T*F*sizeof(float));
    float sc=sqrtf((float)D); for(auto&v:inp)v*=sc; add_posenc(inp,N,F);
    ggml_init_params cp={(size_t)64*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
    ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,N); ggml_set_input(x);
    const bool flash=runtime.accelerated();  // CPU-only runs keep the upstream graph
    ggml_tensor*h=sanm_layer(c,m,"encoder.encoders0.0.",x,N,false,flash);
    for(int i=0;i<m.c.num_blocks-1;i++) h=sanm_layer(c,m,"encoder.encoders."+std::to_string(i)+".",h,N,true,flash);
    h=lnorm(c,h,m.g("encoder.after_norm.weight"),m.g("encoder.after_norm.bias"));
    for(int i=0;i<m.c.tp_blocks;i++) h=sanm_layer(c,m,"encoder.tp_encoders."+std::to_string(i)+".",h,N,true,flash);
    h=lnorm(c,h,m.g("encoder.tp_norm.weight"),m.g("encoder.tp_norm.bias"));
    ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false);
    ggml_tensor*logits=nullptr; std::vector<ggml_tensor*> best;
    if(flash){ best=funasr::chunked_argmax(c,m.g("ctc.ctc_lo.weight"),m.g("ctc.ctc_lo.bias"),h);  // no [V, N] logits
      for(auto t:best) ggml_build_forward_expand(gf,t); }
    else { logits=lin(c,m.g("ctc.ctc_lo.weight"),m.g("ctc.ctc_lo.bias"),h);  // [V, N]
      ggml_set_output(logits); ggml_build_forward_expand(gf,logits); }
    auto sched=runtime.allocate_graph(gf);
    ggml_backend_tensor_set(x,inp.data(),0,ggml_nbytes(x));
    if(ggml_backend_sched_graph_compute(sched,gf)!=GGML_STATUS_SUCCESS){ggml_backend_sched_free(sched);ggml_free(c);throw std::runtime_error("compute failed");}
    std::vector<int> frame_ids;   // greedy CTC: argmax per frame -> collapse -> drop blank
    if(flash) frame_ids=funasr::read_argmax(best);
    else { std::vector<float> lg((size_t)V*N); ggml_backend_tensor_get(logits,lg.data(),0,ggml_nbytes(logits));
      for(int n=0;n<N;n++){ const float*col=&lg[(size_t)n*V]; int am=0; float bv=col[0];
        for(int v=1;v<V;v++) if(col[v]>bv){bv=col[v];am=v;}
        frame_ids.push_back(am); } }
    std::vector<int> seg_ids; int prev=-1;
    for(int am:frame_ids){ if(am!=prev && am!=m.c.blank) seg_ids.push_back(am); prev=am; }

    recognition result; result.ids=std::move(seg_ids);
    result.text=detok_sv(result.ids,vocab,false);
    result.tagged_text=detok_sv(result.ids,vocab,true);
    for(int id:result.ids) if(id>=0 && id<(int)vocab.size()) {
      const auto &piece=vocab[id];
      if(piece.rfind("<|",0)==0) result.raw_tags.push_back(piece);
    }
    ggml_backend_sched_free(sched); ggml_free(c);
    return result;
  }
};
sensevoice_runtime::sensevoice_runtime(const std::string &path,const backend_options &options):state(new impl(path,options)){}
sensevoice_runtime::~sensevoice_runtime()=default;
recognition sensevoice_runtime::features(const std::vector<float> &fb,int frames){return state->features(fb,frames);}
recognition sensevoice_runtime::audio(const std::vector<float> &samples){int frames=0;auto fb=compute_fbank(samples,frames);return features(fb,frames);}
bool sensevoice_runtime::has_vocab()const{return !state->vocab.empty();}
}
