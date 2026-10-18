// CPU scheduling, weights, query dtypes, attention and fail-closed backend selection.
#include "funasr_backend.h"
#include <cmath>
#include <iostream>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        funasr::backend_options options;
        options.threads = 2;
        funasr::backend runtime(options);
        ggml_init_params weights_params = {1024 * 1024, nullptr, false};
        auto weights = ggml_init(weights_params);
        auto w = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 2, 2);
        auto e = ggml_new_tensor_2d(weights, GGML_TYPE_F16, 2, 1);
        const float data[] = {1, 2, 3, 4};
        memcpy(w->data, data, sizeof(data));
        auto half = static_cast<ggml_fp16_t *>(e->data);
        half[0] = ggml_fp32_to_fp16(1.5f); half[1] = ggml_fp32_to_fp16(-2.0f);
        std::map<std::string, ggml_tensor *> tensors = {{"linear.weight", w}, {"embed.weight", e}};
        runtime.prepare_weights(weights, tensors);
        float embedding[2];
        funasr::embedding_row(tensors.at("embed.weight"), 0, 2, embedding);
        require(embedding[0] == 1.5f && embedding[1] == -2.0f, "F16 embedding conversion");
        bool failed = false;
        try { funasr::embedding_row(tensors.at("embed.weight"), 1, 2, embedding); }
        catch (const std::exception &) { failed = true; }
        require(failed, "invalid embedding index accepted");
        require(!runtime.accelerated(), "CPU backend reported an accelerator");
        // Different graph/input shapes must work across successive segment runs.
        for (int frames : {1, 3, 1}) {
            ggml_init_params params = {1024 * 1024, nullptr, true};
            auto ctx = ggml_init(params);
            auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, frames);
            ggml_set_input(x);
            auto y = ggml_mul_mat(ctx, tensors.at("linear.weight"), x);
            ggml_set_output(y);
            auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, y);
            auto sched = runtime.allocate_graph(graph, frames != 3);
            std::vector<float> input(2 * frames, 1.0f), output(2 * frames);
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "compute failed");
            ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
            for (int i = 0; i < frames; ++i)
                require(output[2*i] == 3 && output[2*i+1] == 7, "incorrect graph result");
            ggml_backend_sched_free(sched); ggml_free(ctx);
        }
        // The upstream attention graph and flash attention (F16 keys and values) against a direct
        // computation.
        for (bool flash : {false, true}) for (int frames : {9, 37, 100}) {
            const int D = 8, heads = 2, dk = D / heads;
            std::vector<float> qkv[3];
            for (int t = 0; t < 3; ++t) {
                qkv[t].resize((size_t) D * frames);
                for (size_t i = 0; i < qkv[t].size(); ++i) qkv[t][i] = sinf(0.37f * i + 1.3f * t) * (t + 1);
            }
            ggml_init_params params = {1024 * 1024, nullptr, true};
            auto ctx = ggml_init(params);
            ggml_tensor * in[3];
            for (auto & t : in) { t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, frames); ggml_set_input(t); }
            auto y = funasr::attention(ctx, in[0], in[1], in[2], heads, flash);
            ggml_set_output(y);
            auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, y);
            auto sched = runtime.allocate_graph(graph);
            for (int t = 0; t < 3; ++t) ggml_backend_tensor_set(in[t], qkv[t].data(), 0, qkv[t].size() * sizeof(float));
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "attention compute failed");
            std::vector<float> output((size_t) D * frames);
            ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
            for (int h = 0; h < heads; ++h) for (int i = 0; i < frames; ++i) {
                std::vector<double> p(frames);
                double top = -INFINITY, sum = 0;
                for (int j = 0; j < frames; ++j) {
                    double s = 0;
                    for (int c = 0; c < dk; ++c) s += qkv[0][(size_t) i * D + h * dk + c] * qkv[1][(size_t) j * D + h * dk + c];
                    p[j] = s / sqrt((double) dk); top = std::max(top, p[j]);
                }
                for (auto & v : p) { v = exp(v - top); sum += v; }
                for (int c = 0; c < dk; ++c) {
                    double o = 0;
                    for (int j = 0; j < frames; ++j) o += p[j] / sum * qkv[2][(size_t) j * D + h * dk + c];
                    require(fabs(output[(size_t) i * D + h * dk + c] - o) < (flash ? 1e-2 : 1e-4), "incorrect attention result");
                }
            }
            ggml_backend_sched_free(sched); ggml_free(ctx);
        }
        for (const auto & kind : {"cuda", "hexagon"}) {
            funasr::backend_options invalid; invalid.backend = kind;
            failed = false;
            try { funasr::backend rejected(invalid); } catch (const std::exception &) { failed = true; }
            require(failed, "invalid/unavailable explicit backend fell back silently");
        }
        std::cout << "sensevoice backend tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
