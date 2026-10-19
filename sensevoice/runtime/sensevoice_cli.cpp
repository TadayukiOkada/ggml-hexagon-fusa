#include "sensevoice_runtime.h"
#include "sensevoice_json.h"
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
static const int WINLEN=400;
static int sensevoice_main(int argc,char**argv){
  funasr::backend_options backend_options;
  std::string gguf_path,fbank_path,wav_path,vad_path; int vad_maxseg=30000; bool ids_mode=false,keep_tags=false,json_mode=false;
  for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"-m")&&i+1<argc)gguf_path=argv[++i];
    else if(!strcmp(argv[i],"-f")&&i+1<argc)fbank_path=argv[++i];
    else if(!strcmp(argv[i],"-a")&&i+1<argc)wav_path=argv[++i];
    else if(!strcmp(argv[i],"--vad")&&i+1<argc)vad_path=argv[++i];
    else if(!strcmp(argv[i],"--vad-maxseg")&&i+1<argc)vad_maxseg=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--backend")&&i+1<argc)backend_options.backend=argv[++i];
    else if(!strcmp(argv[i],"--backend-lib")&&i+1<argc)backend_options.library=argv[++i];
    else if(!strcmp(argv[i],"--device")&&i+1<argc)backend_options.device=argv[++i];
    else if(!strcmp(argv[i],"--threads")&&i+1<argc){std::string value=argv[++i];size_t end=0;backend_options.threads=std::stoi(value,&end);if(end!=value.size())throw std::runtime_error("invalid --threads");}
    else if(!strcmp(argv[i],"--trace-schedule"))backend_options.trace=true;
    else if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){fprintf(stdout,"usage: %s -m model.gguf (-a audio.wav | -f fbank.bin) [--backend cpu|hexagon] [--backend-lib path] [--device HTP0] [--threads N] [--trace-schedule] [--vad fsmn-vad.gguf [--vad-maxseg ms]] [--ids] [--keep-tags] [--json]\n",argv[0]);return 0;}
    else if(!strcmp(argv[i],"--json"))json_mode=true;
    else if(!strcmp(argv[i],"--ids"))ids_mode=true;
    else if(!strcmp(argv[i],"--keep-tags"))keep_tags=true;
    else {fprintf(stderr,"usage: %s -m model.gguf (-a audio.wav | -f fbank.bin) [--backend cpu|hexagon] [--backend-lib path] [--device HTP0] [--threads N] [--trace-schedule] [--vad fsmn-vad.gguf [--vad-maxseg ms]] [--ids] [--keep-tags] [--json]\n",argv[0]);return 1;} }
  if(gguf_path.empty()||(fbank_path.empty()&&wav_path.empty())){fprintf(stderr,"missing args\n");return 1;}

  funasr::sensevoice_runtime runtime(gguf_path,backend_options);
  const int F=560;
  auto emit=[&](const funasr::recognition &result){
    if(json_mode){printf("%s\n",funasr::recognition_json(result).c_str());}
    else if(ids_mode || !runtime.has_vocab()){for(int id:result.ids) printf("%d ",id);}
    else {
      printf("%s",(keep_tags?result.tagged_text:result.text).c_str());
    }
  };
  int64_t t0=ggml_time_us();
  if(!vad_path.empty()){
    std::vector<float> wav; if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
    std::vector<std::pair<int,int>> segs;
    if(!funasr_vad_segments(vad_path,wav,vad_maxseg,segs,backend_options.threads)){fprintf(stderr,"vad failed\n");return 1;}
    for(auto&s:segs){ int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
      if(end>(int)wav.size())end=wav.size(); if(end-off<WINLEN)continue;
      std::vector<float> seg(wav.begin()+off,wav.begin()+end); emit(runtime.audio(seg)); }
    fprintf(stderr,"[sensevoice] %zu vad segments\n",segs.size());
  } else {
    int32_t T=0,Fc=F; std::vector<float> fb;
    if(!wav_path.empty()){
      std::vector<float> wav; if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)){fprintf(stderr,"read audio failed\n");return 1;}
      emit(runtime.audio(wav));
    } else {
      FILE*f=fopen(fbank_path.c_str(),"rb"); if(!f){fprintf(stderr,"open fbank\n");return 1;}
      if(fread(&T,4,1,f)!=1||fread(&Fc,4,1,f)!=1){fclose(f);return 1;}
      if(T<1 || Fc!=F){fclose(f);throw std::runtime_error("fbank must have positive frames and 560 features");}
      fb.resize((size_t)T*Fc); if((int)fread(fb.data(),4,fb.size(),f)!=(int)fb.size()){fclose(f);return 1;} fclose(f);
    }
    if(wav_path.empty())emit(runtime.features(fb,T));
  }
  printf("\n");
  fprintf(stderr,"[sensevoice] done %.2fs\n",(ggml_time_us()-t0)/1e6);
  // Weight contexts/buffers are owned by runtime.
  return 0;
}

int main(int argc,char **argv){
  try { return sensevoice_main(argc,argv); }
  catch(const std::exception &error){fprintf(stderr,"sensevoice: %s\n",error.what());return 1;}
}
