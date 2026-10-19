#pragma once
#include "sensevoice_runtime.h"
#include <sstream>
#include <iomanip>
namespace funasr {
inline std::string json_string(const std::string &s){
    std::ostringstream o; o << '"';
    for(unsigned char c:s){switch(c){case '"':o<<"\\\"";break;case '\\':o<<"\\\\";break;case '\n':o<<"\\n";break;case '\r':o<<"\\r";break;case '\t':o<<"\\t";break;
        default:if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<(int)c<<std::dec;else o<<c;}}
    o << '"';return o.str();
}
inline std::string recognition_json(const recognition &r){
    std::string language,emotion,itn;std::vector<std::string> events;
    const std::vector<std::string> languages={"zh","en","yue","ja","ko","nospeech"};
    const std::vector<std::string> emotions={"NEUTRAL","HAPPY","SAD","ANGRY","FEARFUL","DISGUSTED","SURPRISED","EMO_UNKNOWN"};
    const std::vector<std::string> event_names={"Speech","BGM","Applause","Laughter","Cry","Sneeze","Breath","Cough","Sing","Speech_Noise"};
    for(const auto &tag:r.raw_tags){
        if(tag.size()<4)continue;auto v=tag.substr(2,tag.size()-4);
        if(std::find(languages.begin(),languages.end(),v)!=languages.end())language=v;
        else if(std::find(emotions.begin(),emotions.end(),v)!=emotions.end())emotion=v;
        else if(v=="withitn"||v=="woitn")itn=v;
        else if(std::find(event_names.begin(),event_names.end(),v)!=event_names.end())events.push_back(v);
    }
    auto nullable=[](const std::string &s){return s.empty()?std::string("null"):json_string(s);};
    auto array=[](const std::vector<std::string>&v){std::string s="[";for(size_t i=0;i<v.size();i++){if(i)s+=",";s+=json_string(v[i]);}return s+"]";};
    std::string ids="[";for(size_t i=0;i<r.ids.size();i++){if(i)ids+=",";ids+=std::to_string(r.ids[i]);}ids+="]";
    return "{\"text\":"+json_string(r.text)+",\"tagged_text\":"+json_string(r.tagged_text)+",\"ids\":"+ids+",\"analysis\":{\"language\":"+nullable(language)+",\"emotion\":"+nullable(emotion)+",\"events\":"+array(events)+",\"itn\":"+nullable(itn)+",\"raw_tags\":"+array(r.raw_tags)+",\"confidence\":null}}";
}
}
