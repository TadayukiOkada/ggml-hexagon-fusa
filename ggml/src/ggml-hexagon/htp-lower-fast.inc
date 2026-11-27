// Fork-only part of htp-lower.cpp (included there): the transactions of htp_lower_graph(), with the
// same lowered roots and kernel params as upstream's loops and much less work.
//
// Upstream applies a root, then scans the touched nodes in the order they were first touched: a dirty
// node is solved; if that fails, its lowerable outputs and its lowerable selected inputs are lowered
// (an expansion) and the scan starts again from the first node. The transaction is rolled back when a
// scan without an expansion leaves a node unsolved. SenseVoice's 70-layer encoder takes 1.1 s on the
// host CPU this way. Here:
//
// - A visit of a dirty node can do something only if one of its roots was lowered since its last visit
//   (stale: the solve may give another answer, and its outputs or inputs may not be lowerable any more)
//   or one of its lowerable inputs may have become a selected root (woken). The other visits fail
//   without an expansion and change nothing. Upstream's scan therefore does the same solves and
//   expansions as always taking the first touched node that is stale or woken.
// - htp_lower_inplace_types_ok() and htp_kparams_solve() depend only on the node, the lowered flags of
//   its inputs and outputs and its kernel params, so their results are kept for the whole graph.
// - A transaction ends as soon as a dirty node can no longer change, because upstream then rolls it
//   back too: either all its roots are lowered or not lowerable, or none of the nodes that can still
//   act is connected to it through lowerable roots that are not lowered yet (found with cuts in the
//   node order that no such root spans).
//
// GGML_HEXAGON_LOWER_FAST=0 runs upstream's loops, =1 runs this without the connectivity check.
// GGML_HEXAGON_LOWER_VERIFY=1 runs upstream's loops first and aborts if the results differ.

#include <chrono>
#include <climits>
#include <cstdlib>

static int htp_lower_env(const char * name, int def) {
    const char * v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

// Range add and range minimum over the cut positions 0..n-1 (cut k lies between nodes k-1 and k).
struct htp_lower_cut_tree {
    int              n = 1;
    std::vector<int> mn, lz;

    void init(int n_) {
        n = n_ > 0 ? n_ : 1;
        mn.assign(4 * (size_t) n, 0);
        lz.assign(4 * (size_t) n, 0);
    }
    void add(int l, int r, int v) {
        if (l <= r) {
            add(1, 0, n - 1, l, r, v);
        }
    }
    int min(int l, int r) const { return l <= r ? min(1, 0, n - 1, l, r) : INT_MAX / 2; }

  private:
    void add(int x, int lo, int hi, int l, int r, int v) {
        if (r < lo || hi < l) {
            return;
        }
        if (l <= lo && hi <= r) {
            mn[x] += v;
            lz[x] += v;
            return;
        }
        const int m = (lo + hi) / 2;
        add(2 * x, lo, m, l, r, v);
        add(2 * x + 1, m + 1, hi, l, r, v);
        mn[x] = std::min(mn[2 * x], mn[2 * x + 1]) + lz[x];
    }
    int min(int x, int lo, int hi, int l, int r) const {
        if (r < lo || hi < l) {
            return INT_MAX / 2;
        }
        if (l <= lo && hi <= r) {
            return mn[x];
        }
        const int m = (lo + hi) / 2;
        return std::min(min(2 * x, lo, m, l, r), min(2 * x + 1, m + 1, hi, l, r)) + lz[x];
    }
};

template <typename CanLower, typename IsSelected, typename IsSeed, typename TryLowerRoot>
static bool htp_lower_transactions_fast(
    const struct ggml_hexagon_session *                                           sess,
    std::vector<htp_opnode> &                                                     nodes,
    const std::vector<const ggml_tensor *> &                                      roots,
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> &   classes,
    const std::unordered_map<const ggml_tensor *, size_t> &                       owner_idx,
    const std::unordered_map<const ggml_tensor *, std::vector<size_t>> &          root_users,
    CanLower &                                                                    can_lower,
    IsSelected &                                                                  is_selected_root,
    IsSeed &                                                                      is_f16_consumer_seed,
    TryLowerRoot &                                                                try_lower_root,
    size_t &                                                                      n_lowered_roots,
    size_t &                                                                      n_lowered_mm,
    size_t &                                                                      bytes_saved) {
    static const int opt_fast   = htp_lower_env("GGML_HEXAGON_LOWER_FAST", 2);
    static const int opt_verify = htp_lower_env("GGML_HEXAGON_LOWER_VERIFY", 0);
    if (opt_fast <= 0) {
        return false;
    }

    constexpr size_t n_kp = HTP_OP_MAX_KERN_PARAMS;
    const size_t     n_nodes = nodes.size();
    const int        n_roots = (int) roots.size();

    // Upstream's result, for the comparison.
    std::vector<int32_t> ref_kp, init_kp;
    std::vector<uint8_t> ref_flags;
    size_t               ref_counts[3] = {};
    const size_t         init_counts[3] = { n_lowered_roots, n_lowered_mm, bytes_saved };
    auto snapshot_kp = [&](std::vector<int32_t> & kp) {
        kp.resize(n_nodes * n_kp);
        for (size_t i = 0; i < n_nodes; ++i) {
            memcpy(&kp[i * n_kp], nodes[i].kernel_params, sizeof(nodes[i].kernel_params));
        }
    };
    if (opt_verify) {
        snapshot_kp(init_kp);
        for (const ggml_tensor * root : roots) {
            if (can_lower(root) && is_selected_root(root)) {
                try_lower_root(root);
            }
        }
        for (const ggml_tensor * root : roots) {
            if (can_lower(root) && is_f16_consumer_seed(root)) {
                try_lower_root(root);
            }
        }
        snapshot_kp(ref_kp);
        for (const ggml_tensor * root : roots) {
            for (const ggml_tensor * t : classes[root]) {
                ref_flags.push_back(ggml_hexagon_tensor_is_lowered_f16(t));
                ggml_hexagon_tensor_clear_flags(const_cast<ggml_tensor *>(t), HTP_TENSOR_LOWERED_F16);
            }
        }
        for (size_t i = 0; i < n_nodes; ++i) {
            memcpy(nodes[i].kernel_params, &init_kp[i * n_kp], sizeof(nodes[i].kernel_params));
        }
        ref_counts[0] = n_lowered_roots;
        ref_counts[1] = n_lowered_mm;
        ref_counts[2] = bytes_saved;
        n_lowered_roots = init_counts[0];
        n_lowered_mm    = init_counts[1];
        bytes_saved     = init_counts[2];
    }

    const auto t_start = std::chrono::steady_clock::now();

    // Dense root ids. No root is lowered yet, so can_lower() is the part that only depends on the graph.
    std::unordered_map<const ggml_tensor *, int> rid_of;
    rid_of.reserve(roots.size());
    for (int r = 0; r < n_roots; ++r) {
        rid_of.emplace(roots[r], r);
    }
    auto rid = [&](const ggml_tensor * t) -> int {
        if (!t) {
            return -1;
        }
        auto it = rid_of.find(htp_tensor_root(t));
        return it == rid_of.end() ? -1 : it->second;
    };

    static const std::vector<size_t> no_users;
    std::vector<uint8_t>                           lowerable(n_roots), lowered(n_roots, 0);
    std::vector<int8_t>                            selected(n_roots, -1);  // cached is_selected_root(), -1 = unknown
    std::vector<const std::vector<const ggml_tensor *> *> cls(n_roots);
    std::vector<const std::vector<size_t> *>       users(n_roots);
    std::vector<std::vector<int>>                  sel_deps(n_roots);  // roots whose selection reads this root's flag
    for (int r = 0; r < n_roots; ++r) {
        lowerable[r] = can_lower(roots[r]);
        cls[r]       = &classes[roots[r]];
        auto uit     = root_users.find(roots[r]);
        users[r]     = uit == root_users.end() ? &no_users : &uit->second;
    }
    // is_selected_root(w) changes only when an input of a producer of w's class becomes F16, i.e. when
    // the root of that input is lowered.
    for (int w = 0; w < n_roots; ++w) {
        if (!lowerable[w]) {
            continue;
        }
        for (const ggml_tensor * t : *cls[w]) {
            auto oit = owner_idx.find(t);
            if (oit == owner_idx.end()) {
                continue;
            }
            for (const ggml_tensor * x : nodes[oit->second].inputs) {
                const int p = rid(x);
                if (p >= 0 && lowerable[p]) {
                    sel_deps[p].push_back(w);
                }
            }
        }
    }
    std::vector<std::vector<int>> out_rids(n_nodes), in_rids(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
        for (const ggml_tensor * t : nodes[i].outputs) {
            out_rids[i].push_back(rid(t));
        }
        for (const ggml_tensor * t : nodes[i].inputs) {
            in_rids[i].push_back(rid(t));
        }
    }

    // span: for each cut, the lowerable roots not lowered yet whose users lie on both sides of it.
    // root_users lists are in node order.
    htp_lower_cut_tree span;
    span.init((int) n_nodes + 1);
    auto span_add = [&](int r, int v) {
        const std::vector<size_t> & u = *users[r];
        if (!u.empty()) {
            span.add((int) u.front() + 1, (int) u.back(), v);
        }
    };
    for (int r = 0; r < n_roots; ++r) {
        if (lowerable[r]) {
            span_add(r, 1);
        }
    }

    auto is_selected = [&](int r) -> bool {
        if (selected[r] < 0) {
            selected[r] = is_selected_root(roots[r]) ? 1 : 0;
        }
        return selected[r] != 0;
    };

    // Solve results per node: lowered flags of inputs and outputs + incoming kernel params -> result.
    struct solve_memo {
        uint32_t mask;
        bool     ok;
        int32_t  kp_in[n_kp];
        int32_t  kp_out[n_kp];
    };
    static_assert(HTP_OP_MAX_INPUTS + HTP_OP_MAX_OUTPUTS <= 32, "solve memo mask is 32 bits");
    std::vector<std::vector<solve_memo>> memo(n_nodes);
    size_t n_solves = 0, n_memo_hits = 0;
    auto solve = [&](size_t idx) -> bool {
        htp_opnode & node = nodes[idx];
        uint32_t     mask = 0;
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            mask |= (uint32_t) ggml_hexagon_tensor_is_lowered_f16(node.inputs[i]) << i;
        }
        for (size_t i = 0; i < node.outputs.size(); ++i) {
            mask |= (uint32_t) ggml_hexagon_tensor_is_lowered_f16(node.outputs[i]) << (HTP_OP_MAX_INPUTS + i);
        }
        ++n_solves;
        for (const solve_memo & m : memo[idx]) {
            if (m.mask == mask && memcmp(m.kp_in, node.kernel_params, sizeof(m.kp_in)) == 0) {
                ++n_memo_hits;
                if (m.ok) {
                    memcpy(node.kernel_params, m.kp_out, sizeof(m.kp_out));
                }
                return m.ok;
            }
        }
        solve_memo m;
        m.mask = mask;
        memcpy(m.kp_in, node.kernel_params, sizeof(m.kp_in));
        m.ok = htp_lower_inplace_types_ok(node) && htp_kparams_solve(sess, node);
        memcpy(m.kp_out, node.kernel_params, sizeof(m.kp_out));
        memo[idx].push_back(m);
        return m.ok;
    };

    struct touched_node {
        size_t  idx;
        bool    dirty;
        bool    stale;
        bool    woken;
        int32_t kernel_params[n_kp];
    };
    std::vector<touched_node> touched;
    std::vector<int>          touched_pos(n_nodes, -1);
    std::vector<uint64_t>     active;  // touched positions that are dirty and stale or woken
    std::vector<int>          tx_roots;
    size_t                    n_dirty = 0;

    auto set_active = [&](int pos) {
        if ((size_t) pos / 64 >= active.size()) {
            active.resize((size_t) pos / 64 + 1, 0);
        }
        active[pos / 64] |= 1ull << (pos % 64);
    };
    auto first_active = [&]() -> int {
        for (size_t w = 0; w < active.size(); ++w) {
            if (active[w]) {
                return (int) (w * 64 + __builtin_ctzll(active[w]));
            }
        }
        return -1;
    };

    auto set_root_flags = [&](int r, bool on) {
        for (const ggml_tensor * t : *cls[r]) {
            if (on) {
                ggml_hexagon_tensor_set_flags(const_cast<ggml_tensor *>(t), HTP_TENSOR_LOWERED_F16);
            } else {
                ggml_hexagon_tensor_clear_flags(const_cast<ggml_tensor *>(t), HTP_TENSOR_LOWERED_F16);
            }
        }
        lowered[r] = on;
        span_add(r, on ? -1 : 1);
        for (int w : sel_deps[r]) {
            selected[w] = -1;
        }
    };

    auto apply_root = [&](int r) -> bool {
        if (lowered[r]) {
            return false;
        }
        tx_roots.push_back(r);
        set_root_flags(r, true);
        for (size_t idx : *users[r]) {
            int pos = touched_pos[idx];
            if (pos < 0) {
                pos              = (int) touched.size();
                touched_pos[idx] = pos;
                touched_node t{};
                t.idx = idx;
                memcpy(t.kernel_params, nodes[idx].kernel_params, sizeof(t.kernel_params));
                touched.push_back(t);
            }
            if (!touched[pos].dirty) {
                touched[pos].dirty = true;
                ++n_dirty;
            }
            touched[pos].stale = true;
            set_active(pos);
        }
        for (int w : sel_deps[r]) {
            if (lowered[w]) {
                continue;
            }
            for (size_t idx : *users[w]) {
                const int pos = touched_pos[idx];
                if (pos >= 0 && touched[pos].dirty) {
                    touched[pos].woken = true;
                    set_active(pos);
                }
            }
        }
        return true;
    };

    // Whether a dirty node that is not active can never change again: no active node is connected to
    // it through lowerable roots that are not lowered yet. Checked for the first and last such node.
    std::vector<size_t> active_idx;
    auto stuck_for_good = [&]() -> bool {
        size_t n_active = 0;
        for (uint64_t w : active) {
            n_active += (size_t) __builtin_popcountll(w);
        }
        if (n_active == n_dirty || n_active == 0) {
            return false;  // no stuck node, or the transaction ends anyway
        }
        size_t s_min = SIZE_MAX, s_max = 0;
        active_idx.clear();
        for (size_t pos = 0; pos < touched.size(); ++pos) {
            const touched_node & t = touched[pos];
            if (!t.dirty) {
                continue;
            }
            if (pos / 64 < active.size() && ((active[pos / 64] >> (pos % 64)) & 1)) {
                active_idx.push_back(t.idx);
            } else {
                s_min = std::min(s_min, t.idx);
                s_max = std::max(s_max, t.idx);
            }
        }
        if (s_min == SIZE_MAX || active_idx.empty()) {
            return false;
        }
        auto isolated = [&](size_t s) -> bool {
            bool   has_below = false, has_above = false;
            size_t below = 0, above = SIZE_MAX;
            for (size_t a : active_idx) {
                if (a < s) {
                    has_below = true;
                    below     = std::max(below, a);
                } else {
                    has_above = true;
                    above     = std::min(above, a);
                }
            }
            return (!has_below || span.min((int) below + 1, (int) s) == 0) &&
                   (!has_above || span.min((int) s + 1, (int) above) == 0);
        };
        return isolated(s_min) || (s_max != s_min && isolated(s_max));
    };

    size_t n_tx = 0, n_ok = 0, n_visits = 0, n_dead = 0, n_cut = 0, max_touched = 0;

    auto try_lower = [&](int seed) {
        ++n_tx;
        touched.clear();
        active.clear();
        tx_roots.clear();
        n_dirty = 0;

        apply_root(seed);

        bool dead_end = false;
        while (true) {
            const int pos = first_active();
            if (pos < 0) {
                break;
            }
            active[pos / 64] &= ~(1ull << (pos % 64));
            touched_node & t = touched[pos];
            ++n_visits;
            t.woken = false;
            if (t.stale) {
                t.stale = false;
                if (solve(t.idx)) {
                    t.dirty = false;
                    --n_dirty;
                    continue;
                }
            }
            const size_t idx      = t.idx;
            bool         expanded = false;
            bool         waiting  = false;
            for (int r : out_rids[idx]) {
                if (r >= 0 && lowerable[r] && !lowered[r] && apply_root(r)) {
                    expanded = true;
                }
            }
            for (int r : in_rids[idx]) {
                if (r >= 0 && lowerable[r] && !lowered[r]) {
                    if (is_selected(r)) {
                        if (apply_root(r)) {
                            expanded = true;
                        }
                    } else {
                        waiting = true;
                    }
                }
            }
            // All roots of the node are lowered or not lowerable: it stays unsolved.
            if (!expanded && !waiting) {
                dead_end = true;
                ++n_dead;
                break;
            }
            if (opt_fast >= 2 && stuck_for_good()) {
                dead_end = true;
                ++n_cut;
                break;
            }
        }
        max_touched = std::max(max_touched, touched.size());

        for (const touched_node & t : touched) {
            touched_pos[t.idx] = -1;
        }

        if (dead_end || n_dirty > 0) {
            for (int r : tx_roots) {
                set_root_flags(r, false);
            }
            for (const touched_node & t : touched) {
                memcpy(nodes[t.idx].kernel_params, t.kernel_params, sizeof(t.kernel_params));
            }
            return;
        }

        ++n_ok;
        for (int r : tx_roots) {
            ++n_lowered_roots;
            bytes_saved += ggml_nbytes(roots[r]) / 2;
            if (htp_lower_is_mm_root(nodes, owner_idx, roots[r])) {
                ++n_lowered_mm;
            }
        }
    };

    for (int r = 0; r < n_roots; ++r) {
        if (lowerable[r] && !lowered[r] && is_selected(r)) {
            try_lower(r);
        }
    }
    for (int r = 0; r < n_roots; ++r) {
        if (lowerable[r] && !lowered[r] && is_f16_consumer_seed(roots[r])) {
            try_lower(r);
        }
    }

    const double t_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    HEX_VERBOSE("ggml-hex: %s lower-fast mode=%d tx=%zu ok=%zu visits=%zu solves=%zu memo_hits=%zu dead=%zu cut=%zu "
                "max_touched=%zu %.3f ms\n",
                htp_session_name(sess), opt_fast, n_tx, n_ok, n_visits, n_solves, n_memo_hits, n_dead, n_cut,
                max_touched, t_ms);

    if (opt_verify) {
        size_t n_diff = 0, k = 0;
        for (const ggml_tensor * root : roots) {
            for (const ggml_tensor * t : classes[root]) {
                if (ref_flags[k++] != (uint8_t) ggml_hexagon_tensor_is_lowered_f16(t)) {
                    if (n_diff++ < 8) {
                        GGML_LOG_ERROR("ggml-hex: lower verify: %s lowered %d, upstream %d\n", t->name,
                                       (int) ggml_hexagon_tensor_is_lowered_f16(t), (int) ref_flags[k - 1]);
                    }
                }
            }
        }
        for (size_t i = 0; i < n_nodes; ++i) {
            if (memcmp(nodes[i].kernel_params, &ref_kp[i * n_kp], sizeof(nodes[i].kernel_params)) != 0) {
                if (n_diff++ < 8) {
                    GGML_LOG_ERROR("ggml-hex: lower verify: kernel params of node %zu (%s) differ\n", i,
                                   nodes[i].dst() ? nodes[i].dst()->name : "?");
                }
            }
        }
        if (ref_counts[0] != n_lowered_roots || ref_counts[1] != n_lowered_mm || ref_counts[2] != bytes_saved) {
            ++n_diff;
            GGML_LOG_ERROR("ggml-hex: lower verify: roots/mm/bytes %zu/%zu/%zu, upstream %zu/%zu/%zu\n", n_lowered_roots,
                           n_lowered_mm, bytes_saved, ref_counts[0], ref_counts[1], ref_counts[2]);
        }
        if (n_diff) {
            GGML_ABORT("ggml-hex: lower verify: %zu differences from upstream's transactions", n_diff);
        }
        GGML_LOG_INFO("ggml-hex: %s lower verify: same as upstream (roots %zu, %zu nodes)\n", htp_session_name(sess),
                      n_lowered_roots, n_nodes);
    }
    return true;
}
