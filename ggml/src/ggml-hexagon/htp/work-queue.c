#include "work-queue.h"
#include "hex-utils.h"

#include <qurt.h>
#include <qurt_hvx.h>

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HAP_farf.h"

#define LOWEST_USABLE_QURT_PRIO (254)

// internal structure kept in thread-local storage per instance of work queue
typedef struct {
    htp_thread_t th;
    work_queue_t queue;
    atomic_uint  busy;  // inside a job, for work_queue_diag()
} worker_context_t;

struct work_queue_task_s {
    work_queue_func_t func;
    void *            data;
    unsigned int      n_threads;
    atomic_uint       barrier;
    atomic_uint       seq;  // q->seqn after this task's push; seq + 1 (no task of this slot) while it is filled
};

// internal structure kept in thread-local storage per instance of work queue
struct work_queue_s {
    atomic_uint        seqn;      // seqno used to detect new jobs
    atomic_uint        idx_read;  // Updated by producer (pop/reclaim)
    atomic_uint        idx_write; // Updated by producer (push)
    uint32_t           idx_mask;
    uint32_t           capacity;

    qurt_thread_t      thread[HTP_MAX_NTHREADS];   // thread ID's of the workers
    worker_context_t   context[HTP_MAX_NTHREADS];  // worker contexts (0 = main, 1..n_workers = workers)
    void *             stack[HTP_MAX_NTHREADS];    // thread stack pointers
    unsigned int       n_threads;                  // total threads (workers + main)
    unsigned int       n_workers;                  // number of active threads (just workers)

    atomic_uint        active;                     // workers are polling/active
    atomic_bool        killed;                     // threads need to exit
    bool               external_mem;               // memory owned externally

    struct work_queue_task_s queue[] __attribute__((aligned(HEX_L2_LINE_SIZE)));
};

static void work_queue_thread(void * context) {
    worker_context_t * me = (worker_context_t *) context;
    work_queue_t       q  = me->queue;
    unsigned int       id = (unsigned int) (me - q->context);

    FARF(HIGH, "work-queue: thread %u (tid %u) started", id, me->th.tid);

    // Tasks are numbered by q->seqn (one increment per push); task s sits in slot (s - 1) & idx_mask, since
    // idx_write and seqn advance together. done = the last task this worker has looked at.
    unsigned int done = 0;

    while (!atomic_load_explicit(&q->killed, memory_order_relaxed)) {
        unsigned int seqn = atomic_load_explicit(&q->seqn, memory_order_acquire);
        if (seqn == done) {
            if (atomic_load_explicit(&q->active, memory_order_relaxed)) {
                hex_pause();
            } else {
                qurt_hvx_unlock();
                qurt_futex_wait(&q->active, 0);
            }
            continue;
        }

        // Look at each new task exactly once, oldest first. Taking the range from idx_read instead (it moves
        // only when the oldest task completes) let a worker that stalled between reading seqn and idx_read
        // (preempted by a higher-priority thread) run the next task once with the old seqn and again on the
        // next pass: barrier wrapped below zero and the main thread spun forever, holding VTCM/HMX, or it
        // reached zero early and the main thread went on while another worker was still running the task.
        // A task this worker takes part in cannot be overwritten before it runs it (the main thread waits for
        // it); a slot reused by a later task while this worker lagged fails the seq check (seqlock: the
        // producer marks seq before refilling the slot) and that task is taken at its own number.
        for (unsigned int s = done + 1; s != seqn + 1; s++) {
            struct work_queue_task_s * task = &q->queue[(s - 1) & q->idx_mask];
            if (atomic_load_explicit(&task->seq, memory_order_acquire) != s) {
                continue;
            }
            work_queue_func_t func = task->func;
            void *            data = task->data;
            unsigned int      n    = task->n_threads;
            atomic_thread_fence(memory_order_acquire);
            if (atomic_load_explicit(&task->seq, memory_order_relaxed) != s || id >= n) {
                continue;
            }

            atomic_store_explicit(&me->busy, 1, memory_order_relaxed);
            func(&me->th, n, id, data);
            atomic_store_explicit(&me->busy, 0, memory_order_relaxed);

            atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);
        }
        done = seqn;
    }

    FARF(HIGH, "work-queue: thread %u (tid %u) stopped", id, me->th.tid);
}

bool work_queue_run(work_queue_t q, work_queue_func_t func, void * data, unsigned int n) {
    if (n <= 1) {
        func(&q->context[0].th, n, 0, data);
        return true;
    }

    if (n > q->n_threads) {
        FARF(ERROR, "work-queue: invalid number of jobs %u for n-threads %u", n, q->n_threads);
        return false;
    }

    unsigned int ir = atomic_load_explicit(&q->idx_read,  memory_order_relaxed);
    unsigned int iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);

    if (((iw + 1) & q->idx_mask) == ir) {
        FARF(ERROR, "work-queue-push: queue is full\n");
        return false;
    }

    struct work_queue_task_s * task = &q->queue[iw];
    const unsigned int seq = atomic_load_explicit(&q->seqn, memory_order_relaxed) + 1;  // only this thread pushes
    atomic_store_explicit(&task->seq, seq + 1, memory_order_relaxed);  // slot iw only holds tasks seq + k * capacity
    atomic_thread_fence(memory_order_release);
    task->func      = func;
    task->data      = data;
    task->n_threads = n;
    atomic_store_explicit(&task->barrier, n, memory_order_relaxed);
    atomic_store_explicit(&task->seq, seq, memory_order_release);

    atomic_store_explicit(&q->idx_write, (iw + 1) & q->idx_mask, memory_order_relaxed);

    // publish job to workers (already awake and polling)
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);

    // main thread runs job #0
    func(&q->context[0].th, n, 0, data);

    atomic_fetch_sub_explicit(&task->barrier, 1, memory_order_release);

    while (atomic_load_explicit(&task->barrier, memory_order_relaxed) > 0) {
        hex_pause();
    }

    atomic_thread_fence(memory_order_acquire);

    atomic_store_explicit(&q->idx_read, (ir + 1) & q->idx_mask, memory_order_relaxed);

    return true;
}

size_t work_queue_sizeof(uint32_t n_threads, uint32_t capacity, uint32_t stack_size) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    size_t size_stacks = stack_size * n_workers;
    size_t size_q = hex_align_up(sizeof(struct work_queue_s) + capacity * sizeof(struct work_queue_task_s), HEX_L2_LINE_SIZE);
    return size_stacks + size_q;
}

size_t work_queue_alignof(void) {
    return 4096;
}

work_queue_t work_queue_init(void * ptr, uint32_t n_threads, uint32_t capacity, uint32_t stack_size, dma_queue_t * dma, struct htp_thread_trace * trace) {
    capacity = hex_ceil_pow2(capacity);
    uint32_t n_workers = n_threads > 1 ? n_threads - 1 : 0;
    unsigned char * mem_blob = (unsigned char *) ptr;

    work_queue_t q = (work_queue_t) (mem_blob + stack_size * n_workers);
    memset(q, 0, sizeof(struct work_queue_s) + capacity * sizeof(struct work_queue_task_s));

    q->n_threads = n_threads;
    q->n_workers = n_workers;
    q->external_mem = true;
    q->capacity = capacity;

    for (unsigned int i = 0; i < n_threads; i++) {
        htp_thread_init(&q->context[i].th, dma, trace);
        q->context[i].queue = q;
        atomic_init(&q->context[i].busy, 0);
    }

    for (unsigned int i = 0; i < n_workers; i++) {
        q->stack[i]  = mem_blob; mem_blob += stack_size;
        q->thread[i] = 0;
    }

    atomic_init(&q->idx_read,  0);
    atomic_init(&q->idx_write, 0);
    atomic_init(&q->seqn,      0);
    atomic_init(&q->active,    0);
    q->idx_mask  = capacity - 1;
    q->killed    = 0;
    for (int i = 0; i < (int) capacity; i++) {
        atomic_init(&q->queue[i].barrier, 0);
        atomic_init(&q->queue[i].seq, 0);
        q->queue[i].func      = NULL;
        q->queue[i].data      = NULL;
        q->queue[i].n_threads = 0;
    }

    // launch the workers
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);

    for (unsigned int i = 0; i < n_workers; i++) {
        qurt_thread_attr_set_stack_addr(&attr, q->stack[i]);
        qurt_thread_attr_set_stack_size(&attr, stack_size);

        char thread_name[32];
        snprintf(thread_name, sizeof(thread_name), "work-queue:%u", i + 1);
        qurt_thread_attr_set_name(&attr, thread_name);

        // set up priority - by default, match the creating thread's prio
        int prio = qurt_thread_get_priority(qurt_thread_get_id());
        if (prio < 1) {
            prio = 1;
        }
        if (prio > LOWEST_USABLE_QURT_PRIO) {
            prio = LOWEST_USABLE_QURT_PRIO;
        }

        qurt_thread_attr_set_priority(&attr, prio);

        int err = qurt_thread_create(&q->thread[i], &attr, work_queue_thread, (void *) &q->context[i + 1]);
        if (err) {
            FARF(ERROR, "Could not launch worker threads!");
            work_queue_free(q);
            return NULL;
        }
    }

    return q;
}

void work_queue_free(work_queue_t q) {
    if (!q) { return; }

    atomic_store_explicit(&q->killed,   1, memory_order_relaxed);
    atomic_store_explicit(&q->active,   1, memory_order_release);
    atomic_fetch_add_explicit(&q->seqn, 1, memory_order_release);
    qurt_futex_wake(&q->active, q->n_workers);

    for (unsigned int i = 0; i < q->n_workers; i++) {
        if (q->thread[i]) {
            int status;
            (void) qurt_thread_join(q->thread[i], &status);
        }
    }
}

void work_queue_wakeup(work_queue_t q) {
    if (!atomic_load_explicit(&q->active, memory_order_relaxed)) {
        atomic_store_explicit(&q->active, 1, memory_order_release);
        qurt_futex_wake(&q->active, q->n_workers);
    }
}

void work_queue_suspend(work_queue_t q) {
    atomic_store_explicit(&q->active, 0, memory_order_release);
}

bool work_queue_is_active(work_queue_t q) {
    return atomic_load_explicit(&q->active, memory_order_relaxed) != 0;
}

void work_queue_diag(work_queue_t q, uint32_t out[WORK_QUEUE_DIAG_WORDS]) {
    const unsigned int ir = atomic_load_explicit(&q->idx_read, memory_order_relaxed);
    const unsigned int iw = atomic_load_explicit(&q->idx_write, memory_order_relaxed);
    uint32_t busy = 0;
    for (unsigned int i = 1; i <= q->n_workers; i++) {  // context 0 is the main thread
        busy |= (uint32_t) atomic_load_explicit(&q->context[i].busy, memory_order_relaxed) << i;
    }
    out[0] = atomic_load_explicit(&q->active, memory_order_relaxed);
    out[1] = atomic_load_explicit(&q->seqn, memory_order_relaxed);
    out[2] = ir;
    out[3] = iw;
    out[4] = ir != iw ? atomic_load_explicit(&q->queue[ir].barrier, memory_order_relaxed) : 0;  // threads not done
    out[5] = busy;  // bit i: worker i is inside a job
}
