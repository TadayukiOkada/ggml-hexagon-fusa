#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

// Startup-only host diagnostics. Records are API/queue events, not DMA traffic
// or DSP execution timings. Disabled mode performs no I/O or synchronization.
// A failed write stops the trace, never the inference.
struct ggml_hexagon_host_io_trace {
    std::atomic<bool> active { false };
    std::mutex mutex;
    FILE *     file     = nullptr;  // guarded by mutex
    size_t     bytes    = 0;
    uint64_t   sequence = 0;
    static constexpr size_t budget = 128 * 1024 * 1024;

    ~ggml_hexagon_host_io_trace() { if (file) std::fclose(file); }
    bool enabled() const { return active.load(std::memory_order_relaxed); }

    // JSON string body. Each input byte expands to at most 6 bytes.
    static void escape(char * dst, size_t cap, const char * src) {
        size_t n = 0;
        for (const unsigned char * p = (const unsigned char *) src; *p && n + 7 <= cap; ++p) {
            if (*p == '"' || *p == '\\') { dst[n++] = '\\'; dst[n++] = (char) *p; }
            else if (*p < 32) n += (size_t) std::snprintf(dst + n, cap - n, "\\u%04x", (unsigned) *p);
            else dst[n++] = (char) *p;
        }
        dst[n] = 0;
    }

    void stop_locked(const char * why) {
        GGML_LOG_WARN("ggml-hex: host I/O trace stopped: %s\n", why);
        std::fclose(file);
        file = nullptr;
        active.store(false, std::memory_order_relaxed);
    }

    void put_locked(const char * line, size_t n) {
        if (std::fwrite(line, 1, n, file) != n || std::fflush(file) != 0) stop_locked(std::strerror(errno));
    }

    // `fields` is a printf format for the record's trailing JSON members (",\"k\":v...").
    GGML_ATTRIBUTE_FORMAT(4, 5)  // `this` is argument 1
    void record(const char * event, const char * device, const char * fields, ...) {
        if (!enabled()) return;
        char tail[768];
        va_list ap;
        va_start(ap, fields);
        const int nt = std::vsnprintf(tail, sizeof(tail), fields, ap);
        va_end(ap);
        char dev[64];
        escape(dev, sizeof(dev), device);

        std::lock_guard<std::mutex> lock(mutex);
        if (!file) return;
        if (nt < 0 || (size_t) nt >= sizeof(tail)) { stop_locked("record too long"); return; }
        const long long now = (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        char line[1024];
        const int n = std::snprintf(line, sizeof(line),
            "{\"version\":1,\"sequence\":%" PRIu64 ",\"monotonic_ns\":%lld,\"event\":\"%s\",\"device\":\"%s\"%s}\n",
            ++sequence, now, event, dev, tail);
        if (n < 0 || (size_t) n >= sizeof(line)) { stop_locked("record too long"); return; }
        if ((size_t) n > budget - bytes) {
            static const char trunc[] = "{\"version\":1,\"event\":\"truncated\",\"budget_bytes\":134217728}\n";
            put_locked(trunc, sizeof(trunc) - 1);
            if (file) stop_locked("budget exhausted");
            return;
        }
        bytes += (size_t) n;
        put_locked(line, (size_t) n);
    }

    void init() {
        const char * path = std::getenv("GGML_HEXAGON_IO_TRACE");
        if (!path || !*path) return;
        // Exclusive creation protects evidence from a later process.
        file = std::fopen(path, "wx");
        if (!file) {
            GGML_LOG_WARN("ggml-hex: cannot create host I/O trace %s: %s\n", path, std::strerror(errno));
            return;
        }
        active.store(true, std::memory_order_relaxed);
        record("schema", "", ",\"budget_bytes\":134217728,"
               "\"scope\":\"hexagon host handlers and queue; excludes CPU-only copies and physical DMA\"");
    }

    void tensor(const char * event, const char * device, const ggml_tensor * t, ggml_backend_buffer_t buffer,
                size_t size, size_t copies, bool repack, bool output_staging = false) {
        if (!enabled()) return;
        char name[6 * GGML_MAX_NAME + 1];
        escape(name, sizeof(name), t->name);
        const uint64_t logical  = (uint64_t) size * copies;
        const bool     measured = !repack || output_staging;
        char calls[24] = "null";
        char moved[24] = "null";
        if (measured) {
            std::snprintf(calls, sizeof(calls), "%zu", copies);
            std::snprintf(moved, sizeof(moved), "%" PRIu64, logical);
        }
        record(event, device,
               ",\"tensor\":\"%s\",\"dtype\":\"%s\",\"usage\":%d,\"handler_logical_bytes\":%" PRIu64
               ",\"host_memcpy_calls\":%s,\"host_memcpy_bytes\":%s,\"repack\":%s,\"memcpy_scope\":\"%s\"",
               name, ggml_type_name(t->type), (int) ggml_backend_buffer_get_usage(buffer), logical, calls, moved,
               repack ? "true" : "false",
               output_staging ? "2d output staging only" : (repack ? "unmeasured repack internals" : "completed plain host copy"));
    }
};

static ggml_hexagon_host_io_trace hex_io_trace;
