#include "sensevoice_json.h"
#include <stdexcept>
int main(){
    funasr::recognition r;
    r.text="a\"\n\\b";
    auto empty=funasr::recognition_json(r);
    if(empty.find("\"emotion\":null")==std::string::npos || empty.find("a\\\"\\n\\\\b")==std::string::npos)throw std::runtime_error("empty tags or escaping");
    r.raw_tags={"<|ja|>","<|HAPPY|>","<|Laughter|>","<|Applause|>","<|withitn|>","<|unseen|>"};
    auto full=funasr::recognition_json(r);
    if(full.find("\"events\":[\"Laughter\",\"Applause\"]")==std::string::npos || full.find("<|unseen|>")==std::string::npos || full.find("\"language\":\"ja\"")==std::string::npos)throw std::runtime_error("lost decoded tags");
}
