// Shared SenseVoice frontend, SAN-M scheduler and greedy CTC runtime.
#pragma once
#include "funasr_backend.h"
#include <memory>
namespace funasr {
struct recognition { std::vector<int> ids; std::string text; std::string tagged_text; std::vector<std::string> raw_tags; };
class sensevoice_runtime {
    struct impl; std::unique_ptr<impl> state;
public:
    sensevoice_runtime(const std::string &model_path, const backend_options &options);
    ~sensevoice_runtime();
    recognition features(const std::vector<float> &fbank, int frames);
    recognition audio(const std::vector<float> &samples);
    bool has_vocab() const;
};
}
