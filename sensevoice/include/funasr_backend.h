// SenseVoice backend integration for ggml-hexagon-fusa (MIT).
#pragma once
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace funasr {
struct backend_options {
    std::string backend = "cpu";
    std::string library;
    std::string device = "HTP0";
    int threads = 8;
    bool trace = false;
};

class backend {
    ggml_backend_t cpu_ = nullptr;
    ggml_backend_t accelerator_ = nullptr;
    ggml_backend_reg_t registry_ = nullptr;
    std::vector<ggml_backend_buffer_t> weights_;
    std::vector<ggml_context *> contexts_;
    backend_options options_;

    void release() {
        for (auto buffer : weights_) ggml_backend_buffer_free(buffer);
        for (auto ctx : contexts_) ggml_free(ctx);
        if (accelerator_) ggml_backend_free(accelerator_);
        if (cpu_) ggml_backend_free(cpu_);
        // The registry is process-global; unload only after all ggml objects die.
    }
    ggml_context * weight_context(size_t count) {
        ggml_init_params params = {ggml_tensor_overhead() * (count + 1), nullptr, true};
        auto ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("cannot allocate weight metadata");
        contexts_.push_back(ctx);
        return ctx;
    }
    void allocate(ggml_context * ctx, ggml_backend_t device) {
        if (!ggml_get_first_tensor(ctx)) return;
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, device);
        if (!buffer) throw std::runtime_error("cannot allocate backend weights");
        weights_.push_back(buffer);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
public:
    explicit backend(const backend_options & options) : options_(options) {
        try {
            if (options.backend != "cpu" && options.backend != "hexagon")
                throw std::runtime_error("--backend must be cpu or hexagon");
            if (options.threads <= 0) throw std::runtime_error("--threads must be positive");
            cpu_ = ggml_backend_cpu_init();
            if (!cpu_) throw std::runtime_error("cannot initialize CPU backend");
            ggml_backend_cpu_set_n_threads(cpu_, options.threads);
            if (options.backend == "hexagon") {
                if (options.library.empty())
                    throw std::runtime_error("--backend hexagon requires --backend-lib");
                registry_ = ggml_backend_load(options.library.c_str());
                if (!registry_ || std::string(ggml_backend_reg_name(registry_)) != "HTP")
                    throw std::runtime_error("cannot load an HTP registry from --backend-lib");
                for (size_t i = 0; i < ggml_backend_reg_dev_count(registry_); ++i) {
                    auto dev = ggml_backend_reg_dev_get(registry_, i);
                    if (options.device == ggml_backend_dev_name(dev)) {
                        accelerator_ = ggml_backend_dev_init(dev, nullptr);
                        break;
                    }
                }
                if (!accelerator_) throw std::runtime_error("Hexagon device unavailable: " + options.device);
            } else if (!options.library.empty()) {
                throw std::runtime_error("--backend-lib requires --backend hexagon");
            }
            fprintf(stderr, "[sensevoice] backend=%s device=%s threads=%d\n",
                    options.backend.c_str(), accelerator_ ? options.device.c_str() : "CPU", options.threads);
        } catch (...) { release(); throw; }
    }
    ~backend() { release(); }
    backend(const backend &) = delete;
    backend & operator=(const backend &) = delete;

    // SAN-M attention multiplies activations by activations, which the backend's per-op budget
    // does not estimate. On v73 its score matmul takes about 10 ms for 512 frames and grows
    // with the square of the length (38 ms for 1024, over the FuSa self-test period), so longer
    // segments (more than about 30 s of audio; --vad keeps them shorter) are computed on the CPU.
    static constexpr int max_accelerated_frames = 512;

    bool accelerates(int frames) const {
        if (accelerator_ && frames > max_accelerated_frames)
            fprintf(stderr, "[sensevoice] %d frames > %d: segment computed on the CPU (use --vad)\n",
                    frames, max_accelerated_frames);
        return accelerator_ && frames <= max_accelerated_frames;
    }

    // Weights get persistent storage on the accelerator, so a graph has no CPU splits for
    // biases, norms or FSMN taps. Query embeddings are read on the host and stay on the CPU.
    void prepare_weights(ggml_context * & source, std::map<std::string, ggml_tensor *> & tensors) {
        auto host = weight_context(tensors.size());
        auto device = accelerator_ ? weight_context(tensors.size()) : nullptr;
        std::map<std::string, ggml_tensor *> placed;
        for (const auto & entry : tensors) {
            auto tensor = ggml_dup_tensor(device && entry.first != "embed.weight" ? device : host, entry.second);
            ggml_set_name(tensor, entry.first.c_str());
            placed[entry.first] = tensor;
        }
        allocate(host, cpu_);
        if (device) allocate(device, accelerator_);
        for (const auto & entry : tensors)
            ggml_backend_tensor_set(placed.at(entry.first), entry.second->data, 0, ggml_nbytes(entry.second));
        tensors = std::move(placed);
        ggml_free(source);
        source = host; // owned by this backend, not by the model map
    }

    // accelerate = false computes every node on the CPU; the accelerator stays in the
    // scheduler because it holds the weights.
    ggml_backend_sched_t allocate_graph(ggml_cgraph * graph, bool accelerate = true) {
        ggml_backend_t devices[] = {accelerator_, cpu_};
        auto sched = ggml_backend_sched_new(accelerator_ ? devices : devices + 1,
                                            nullptr, accelerator_ ? 2 : 1, 32768, false, true);
        if (!sched) throw std::runtime_error("cannot create backend scheduler");
        try {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                    node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
                auto selected = accelerate && accelerator_ && ggml_backend_supports_op(accelerator_, node) ? accelerator_ : cpu_;
                ggml_backend_sched_set_tensor_backend(sched, node, selected);
            }
            if (!ggml_backend_sched_alloc_graph(sched, graph))
                throw std::runtime_error("cannot allocate scheduled graph");
            if (options_.trace) {
                using reason_fn = const char * (*)(ggml_backend_dev_t, const ggml_tensor *);
                auto reason = registry_ ? reinterpret_cast<reason_fn>(ggml_backend_reg_get_proc_address(
                    registry_, "ggml_backend_hexagon_support_reason")) : nullptr;
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    auto node = ggml_graph_node(graph, i);
                    auto assigned = ggml_backend_sched_get_tensor_backend(sched, node);
                    fprintf(stderr, "[sensevoice-schedule] op=%s name=%s backend=%s reason=%s\n",
                            ggml_op_name(node->op), node->name,
                            assigned ? ggml_backend_name(assigned) : "view",
                            accelerator_ && !accelerate ? "frame_limit" :
                            reason ? reason(ggml_backend_get_device(accelerator_), node) : "cpu");
                }
            }
        } catch (...) { ggml_backend_sched_free(sched); throw; }
        return sched;
    }
};

inline void embedding_row(const ggml_tensor * embedding, int id, int width, float * output) {
    if (id < 0 || id >= embedding->ne[1] || embedding->ne[0] != width)
        throw std::runtime_error("invalid query embedding shape/index");
    if (embedding->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(embedding, output, id * embedding->nb[1], width * sizeof(float));
    } else if (embedding->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> row(width);
        ggml_backend_tensor_get(embedding, row.data(), id * embedding->nb[1], width * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(row.data(), output, width);
    } else {
        throw std::runtime_error("query embedding must be F32 or F16");
    }
}

// The HTP softmax takes rows of at most 32 or a multiple of 32. On the accelerator, longer
// attention pads the keys with zero rows to that width, and the mask's -inf columns give them
// zero weight. Segments computed on the CPU keep the upstream graph, and its results.
inline int attention_keys(int frames) { return frames > 32 ? (frames + 31) / 32 * 32 : frames; }

// [keys, frames] mask input, or nullptr when no keys are added.
inline ggml_tensor * attention_mask(ggml_context * ctx, int frames) {
    const int keys = attention_keys(frames);
    if (keys == frames) return nullptr;
    auto mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, keys, frames);
    ggml_set_input(mask);
    return mask;
}

inline void set_attention_mask(ggml_tensor * mask) {
    if (!mask) return;
    const int keys = mask->ne[0], frames = mask->ne[1];
    std::vector<float> data((size_t) keys * frames, 0.0f);
    for (int r = 0; r < frames; ++r)
        std::fill(data.begin() + (size_t) r * keys + frames, data.begin() + (size_t) (r + 1) * keys, -INFINITY);
    ggml_backend_tensor_set(mask, data.data(), 0, ggml_nbytes(mask));
}

// Multi-head attention of contiguous q, k, v [D, T] -> [D, T], as in SenseVoice's SAN-M.
// Without a mask this is the upstream graph; with one, keys and values get zero rows.
inline ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                               int heads, ggml_tensor * mask) {
    const int D = q->ne[0], T = q->ne[1], dk = D / heads, keys = mask ? mask->ne[0] : T;
    const float scale = 1.0f / sqrtf((float) dk);
    if (keys != T) {
        k = ggml_pad(c, k, 0, keys - T, 0, 0);
        v = ggml_pad(c, v, 0, keys - T, 0, 0);
    }
    q = ggml_permute(c, ggml_reshape_3d(c, q, dk, heads, T), 0, 2, 1, 3);
    k = ggml_permute(c, ggml_reshape_3d(c, k, dk, heads, keys), 0, 2, 1, 3);
    auto vh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, dk, heads, keys), 1, 2, 0, 3));
    auto kq = mask ? ggml_soft_max_ext(c, ggml_mul_mat(c, k, q), mask, scale, 0.0f)
                   : ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, k, q), scale));
    return ggml_cont_2d(c, ggml_permute(c, ggml_mul_mat(c, vh, kq), 0, 2, 1, 3), D, T);
}
} // namespace funasr
