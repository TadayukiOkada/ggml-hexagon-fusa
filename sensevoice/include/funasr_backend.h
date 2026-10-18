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

    bool accelerated() const { return accelerator_ != nullptr; }

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
                            accelerator_ && !accelerate ? "cpu_requested" :
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

// Multi-head attention of contiguous q, k, v [D, T] -> [D, T], as in SenseVoice's SAN-M.
// flash = false is the upstream graph. flash = true uses flash attention with F16 keys and
// values: on the HTP it releases VTCM between output blocks, so the FuSa self-test period does
// not bound the segment length, and the T x T scores are never stored.
inline ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                               int heads, bool flash) {
    const int D = q->ne[0], T = q->ne[1], dk = D / heads;
    const float scale = 1.0f / sqrtf((float) dk);
    q = ggml_permute(c, ggml_reshape_3d(c, q, dk, heads, T), 0, 2, 1, 3);
    if (flash) {
        k = ggml_permute(c, ggml_reshape_3d(c, ggml_cast(c, k, GGML_TYPE_F16), dk, heads, T), 0, 2, 1, 3);
        v = ggml_permute(c, ggml_reshape_3d(c, ggml_cast(c, v, GGML_TYPE_F16), dk, heads, T), 0, 2, 1, 3);
        auto o = ggml_flash_attn_ext(c, q, k, v, nullptr, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        return ggml_reshape_2d(c, o, D, T);  // [dk, heads, T]
    }
    k = ggml_permute(c, ggml_reshape_3d(c, k, dk, heads, T), 0, 2, 1, 3);
    auto vh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, dk, heads, T), 1, 2, 0, 3));
    auto kq = ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, k, q), scale));
    return ggml_cont_2d(c, ggml_permute(c, ggml_mul_mat(c, vh, kq), 0, 2, 1, 3), D, T);
}
// Per-frame argmax of a linear layer (weights w, optional bias b) over x [D, N], computed in
// chunks of frames so that the [V, N] scores are never stored at once: greedy CTC needs only the
// best token of each frame, and the CTC logits are 25055 x frames floats (1 GB for 10 minutes).
// A chunk also bounds the HTP argmax, which does not release VTCM inside the op: 256 frames of
// 25055 scores take about 9 ms on v73 (512 frames 17 ms).
// Returns the I32 argmax outputs of the chunks in frame order; build the graph from all of them.
static constexpr int argmax_chunk_frames = 256;

inline std::vector<ggml_tensor *> chunked_argmax(ggml_context * c, ggml_tensor * w, ggml_tensor * b,
                                                 ggml_tensor * x, int chunk = argmax_chunk_frames) {
    std::vector<ggml_tensor *> out;
    for (int64_t i = 0; i < x->ne[1]; i += chunk) {
        const int64_t n = std::min<int64_t>(chunk, x->ne[1] - i);
        auto y = ggml_mul_mat(c, w, ggml_view_2d(c, x, x->ne[0], n, x->nb[1], i * x->nb[1]));
        auto am = ggml_argmax(c, b ? ggml_add(c, y, b) : y);
        ggml_set_output(am);
        out.push_back(am);
    }
    return out;
}

inline std::vector<int> read_argmax(const std::vector<ggml_tensor *> & chunks) {
    std::vector<int> ids;
    for (auto t : chunks) {
        std::vector<int32_t> part(t->ne[0]);
        ggml_backend_tensor_get(t, part.data(), 0, ggml_nbytes(t));
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}
} // namespace funasr
