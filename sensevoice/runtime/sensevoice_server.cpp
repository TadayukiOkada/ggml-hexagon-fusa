// Local persistent server; shares all inference with the CLI.
#include "sensevoice_runtime.h"
#include "sensevoice_json.h"
#include "json.h"
#include <httplib.h>
#include <atomic>
#include <chrono>
#include <mutex>
using nemo_speech::json::Value;
static const size_t MAX_BYTES=16000*2*30;
static std::vector<float> samples(const std::string &pcm){
    if(pcm.size()%2||pcm.size()>MAX_BYTES)throw std::invalid_argument("16k mono PCM16, maximum 30s required");
    std::vector<float> v(pcm.size()/2);
    for(size_t i=0;i<v.size();i++){uint16_t u=(uint8_t)pcm[2*i]|((uint16_t)(uint8_t)pcm[2*i+1]<<8);v[i]=(int16_t)u/32768.f;}
    return v;
}
int main(int argc,char **argv){try{
    std::string path;int port=8767;funasr::backend_options options;
    for(int i=1;i<argc;i++){
        std::string arg=argv[i];
        if(arg=="--help"){printf("sensevoice-server -m MODEL [--port N] [--backend cpu|hexagon] [--backend-lib PATH] [--device HTP1] [--threads N]\n");return 0;}
        if(i+1==argc)throw std::invalid_argument("missing option value");
        std::string value=argv[++i];
        if(arg=="-m")path=value;else if(arg=="--port")port=std::stoi(value);
        else if(arg=="--backend")options.backend=value;else if(arg=="--backend-lib")options.library=value;
        else if(arg=="--device")options.device=value;else if(arg=="--threads")options.threads=std::stoi(value);
        else throw std::invalid_argument("unknown option: "+arg);
    }
    if(path.empty()||port<1||port>65535)throw std::invalid_argument("model and valid port required");
    funasr::sensevoice_runtime runtime(path,options);
    std::mutex slot;httplib::Server server;server.new_task_queue=[] { return new httplib::ThreadPool(4); };server.set_payload_max_length(MAX_BYTES+4096);
    server.set_read_timeout(40);server.set_write_timeout(40);
    server.Get("/ready",[&](const httplib::Request&,httplib::Response &res){
        Value root(Value::Object{});root["ready"]=true;root["backend"]=options.backend;root["device"]=options.backend=="cpu"?"CPU":options.device;root["model"]=path;root["is_mock"]=false;
        res.set_content(root.dump(),"application/json");
    });
    auto infer=[&](const std::string &pcm){
        auto begin=std::chrono::steady_clock::now();auto result=runtime.audio(samples(pcm));
        auto response=Value::parse(funasr::recognition_json(result));
        response["compute_ms"]=(double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-begin).count();
        response["samples"]=(double)(pcm.size()/2);response["is_mock"]=false;return response;
    };
    server.Post("/api/transcribe",[&](const httplib::Request &req,httplib::Response &res){
        std::unique_lock<std::mutex> lock(slot,std::try_to_lock);
        if(!lock.owns_lock()){res.status=503;res.set_content("{\"error\":\"busy\"}","application/json");return;}
        try{res.set_content(infer(req.body).dump(),"application/json");}
        catch(const std::exception &e){res.status=400;res.set_content("{\"error\":"+funasr::json_string(e.what())+"}","application/json");}
    });
    server.WebSocket("/api/session",[&](const httplib::Request&,httplib::ws::WebSocket &socket){
        std::unique_lock<std::mutex> lock(slot,std::try_to_lock);
        if(!lock.owns_lock()){socket.send("{\"type\":\"error\",\"code\":\"busy\"}");return;}
        socket.send("{\"type\":\"session.ready\",\"sample_rate\":16000,\"is_mock\":false}");
        std::string pcm,message;
        while(true){const auto kind=socket.read(message);if(kind==httplib::ws::ReadResult::Fail)break;
            try{
                if(kind==httplib::ws::ReadResult::Binary){
                    if(message.size()%2||message.size()>3200||message.size()>MAX_BYTES-pcm.size())throw std::invalid_argument("PCM size limit");
                    pcm+=message;continue;
                }
                const auto event=Value::parse(message);const auto type=event.string_or("type");
                if(type=="cancel"){pcm.clear();socket.send("{\"type\":\"session.cancelled\"}");break;}
                if(type!="stop")throw std::invalid_argument("unsupported control");
                auto result=infer(pcm);result["type"]="result.final";result["segment_id"]=1;result["revision"]=1;
                if(!socket.send(result.dump()))break;
                socket.send("{\"type\":\"session.done\"}");break;
            }catch(const std::exception &e){socket.send("{\"type\":\"error\",\"message\":"+funasr::json_string(e.what())+"}");break;}
        }
    });
    fprintf(stderr,"SenseVoice ready at http://127.0.0.1:%d\n",port);
    if(!server.listen("127.0.0.1",port))throw std::runtime_error("listen failed");
    return 0;
}catch(const std::exception &e){fprintf(stderr,"sensevoice-server: %s\n",e.what());return 1;}}
