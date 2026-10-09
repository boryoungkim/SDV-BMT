// SDV-BMT native runner unit tests: realtime loop semantics on a simulated clock, event serialization,
// canonical JSON and SHA-256.  Prints one line per test; exit status 0 only if every test passed.
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/events.hpp"
#include "common/json.hpp"
#include "common/loop.hpp"
#include "common/output_pipeline.hpp"
#include "common/reference_output.hpp"
#include "common/sha256.hpp"

using namespace sdv;

static Json reference_spec() {
    Json s = Json::object();
    s["kind"] = "reference_convnet_v1";
    s["task"] = "reference-conformance";
    return s;
}

static int failures = 0;
static int checks = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++checks;                                                                     \
        if (!(cond)) {                                                                \
            ++failures;                                                               \
            std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                             \
    } while (0)

constexpr int64_t MS = 1000000;

// Every read advances time by `step`; `jumps` moves time forward at a given read count (preemption).
struct SimClock {
    int64_t t = 0;
    int64_t step = 1000;  // 1 us per read
    int64_t reads = 0;
    std::map<int64_t, int64_t> jumps;  // read index -> new absolute time
    int64_t now() {
        auto it = jumps.find(reads++);
        if (it != jumps.end() && it->second > t) t = it->second;
        int64_t v = t;
        t += step;
        return v;
    }
    int64_t peek() const { return t; }
};

struct SimBackend {
    SimClock& clock;
    std::function<int64_t(int64_t item)> service;  // device time per item
    std::vector<int64_t> item_of_slot;
    std::set<int64_t> submit_errors, run_errors, bad_output;
    int64_t done_at = 0, item = 0;
    std::vector<std::pair<int64_t, std::string>> marks;
    int64_t submit_cost = 0;

    bool submit(int64_t, int slot, ErrorInfo& err) {
        item = item_of_slot.at(size_t(slot));
        if (submit_errors.count(item)) {
            err = ErrorInfo{"sim_submit_error", "injected"};
            return false;
        }
        clock.t += submit_cost;
        done_at = clock.peek() + service(item);
        return true;
    }
    int poll(ErrorInfo& err) {
        if (clock.peek() < done_at) return 0;
        if (run_errors.count(item)) {
            err = ErrorInfo{"sim_run_error", "injected"};
            return -1;
        }
        return 1;
    }
    char raw_buf[80] = {};
    bool collect(DeviceFacts& f, std::string_view& raw, ErrorInfo&) {
        f.timing_method = "sim:device";
        f.segment("device_execution", service(item));
        f.observe("sim.item", item);
        int64_t logits[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        logits[item % 10] = 7;
        std::memcpy(raw_buf, logits, sizeof logits);
        raw = std::string_view(raw_buf, bad_output.count(item) ? 79 : 80);
        return true;
    }
    void mark_admission(int64_t rid, const char* what) { marks.emplace_back(rid, what); }
};

struct Run {
    SimClock clock;
    SimBackend be{clock, [](int64_t) { return 3 * MS; }, {}, {}, {}, {}, 0, 0, {}, 0};
    OutputPipeline out{reference_spec()};
    LoopConfig cfg;
    std::vector<PlannedItem> items;

    void plan(int n, int64_t period, std::set<int64_t> failed = {}) {
        cfg.replay_mode = period ? "fixed_rate" : "asap";
        cfg.asap = period == 0;
        cfg.lead_ns = 1 * MS;
        if (cfg.request_timeout_ns == 0) cfg.request_timeout_ns = 1000 * MS;
        if (cfg.drain_ns == 0) cfg.drain_ns = 1000 * MS;
        int slot = 0;
        for (int i = 0; i < n; ++i) {
            if (period) cfg.offsets_ns.push_back(i * period);
            PlannedItem p;
            p.item_index = i;
            if (failed.count(i)) {
                p.failure_stage = "preprocess";
            } else {
                p.slot = slot++;
                be.item_of_slot.push_back(i);
            }
            items.push_back(p);
        }
    }
};

using Loop = RealtimeLoop<SimClock, SimBackend, OutputPipeline>;

static std::vector<State> states(const Loop& l) {
    std::vector<State> s;
    for (const auto& f : l.facts()) s.push_back(f.state);
    return s;
}

static std::vector<Json> serialize(const Loop& l, int64_t seq0 = 5, int64_t t0 = 0) {
    EventWriter w(seq0, t0);
    EventContext ctx;
    ctx.codec = "canonical-json-utf8/1";
    write_loop_events(l, w, ctx);
    std::vector<Json> out;
    size_t start = 0;
    const std::string& text = w.text();
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        out.push_back(Json::parse(text.substr(start, end - start)));
        start = end + 1;
    }
    return out;
}

static void check_events_well_formed(const std::vector<Json>& ev, int64_t seq0) {
    int64_t last = -1;
    for (size_t k = 0; k < ev.size(); ++k) {
        CHECK(ev[k].at("seq").as_int() == seq0 + int64_t(k));
        CHECK(ev[k].at("t_run_ns").as_int() >= last);
        CHECK(ev[k].at("schema_version").as_int() == 2);
        CHECK(ev[k].at("run_clock_domain").as_string() == "monotonic");
        last = ev[k].at("t_run_ns").as_int();
    }
}

static void test_all_complete() {
    Run r;
    r.plan(5, 10 * MS);
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    LoopSummary s = l.run();
    CHECK(!s.aborted);
    for (State st : states(l)) CHECK(st == State::Completed);
    for (const auto& f : l.facts()) {
        CHECK(f.t_issue >= f.t_sched);
        CHECK(f.t_submit_begin >= f.t_issue);
        CHECK(f.t_complete - f.t_submit_begin >= 3 * MS);
        CHECK(f.t_complete - f.t_submit_begin < 3 * MS + 20000);
        CHECK(f.t_output_ready >= f.t_complete);
        CHECK(f.out_length > 0 && f.out_sha256.size() == 64);
    }
    CHECK(!s.cap_reached && s.outstanding.empty());
    auto ev = serialize(l);
    check_events_well_formed(ev, 5);
    std::vector<std::string> kinds;
    for (const Json& e : ev) {
        if (e.has("request_id") && e.at("request_id").as_int() == 0) kinds.push_back(e.at("kind").as_string());
    }
    CHECK((kinds == std::vector<std::string>{"issued", "submit_begin", "submit_end", "completed", "device_observation"}));
    CHECK(ev.front().at("phase").as_string() == "measure" && ev.front().at("edge").as_string() == "begin");
    CHECK(ev.back().at("phase").as_string() == "drain" && ev.back().at("edge").as_string() == "end");
    std::printf("ok all_complete\n");
}

static void test_drop_new() {
    Run r;
    r.plan(6, 2 * MS);  // service 3 ms > period 2 ms
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    l.run();
    auto st = states(l);
    CHECK((st == std::vector<State>{State::Completed, State::Dropped, State::Completed, State::Dropped,
                                    State::Completed, State::Dropped}));
    CHECK(r.be.marks.size() == 3 && r.be.marks[0].second == "dropped" && r.be.marks[0].first == 1);
    std::printf("ok drop_new\n");
}

static void test_timeout_late_completion_holds_slot() {
    Run r;
    r.cfg.request_timeout_ns = 2 * MS;
    r.cfg.drain_ns = 100 * MS;
    r.be.service = [](int64_t) { return 5 * MS; };
    r.plan(4, 3 * MS);  // r0 times out at 2 ms, completes at 5 ms; r1 at 3 ms finds the slot held
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    l.run();
    const auto& f = l.facts();
    CHECK(f[0].state == State::TimedOut && f[0].late && f[0].late_ok);
    CHECK(f[0].t_timed_out - f[0].t_submit_begin >= 2 * MS);
    CHECK(f[0].t_timed_out - f[0].t_submit_begin < 2 * MS + 20000);  // timer = t_submit_begin + timeout
    CHECK(f[1].state == State::Dropped);
    CHECK(f[2].state == State::TimedOut && f[2].late);
    auto ev = serialize(l);
    check_events_well_formed(ev, 5);
    int late = 0, timed = 0;
    for (const Json& e : ev) {
        late += e.at("kind").as_string() == "late_completion";
        timed += e.at("kind").as_string() == "timed_out";
        if (e.at("kind").as_string() == "late_completion") CHECK(e.at("result").as_string() == "ok" && e.has("output_ref"));
    }
    CHECK(late == timed && timed >= 2);
    std::printf("ok timeout_late_completion_holds_slot\n");
}

static void test_inclusive_drain_cap() {
    Run r;
    r.cfg.request_timeout_ns = 1 * MS;
    r.cfg.drain_ns = 1 * MS;  // drain == timeout
    r.be.service = [](int64_t) { return 5 * MS; };
    r.plan(2, 10 * MS);
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    LoopSummary s = l.run();
    const auto& f = l.facts();
    CHECK(f[0].state == State::TimedOut && f[0].late);
    CHECK(f[1].state == State::TimedOut && !f[1].late);  // terminal at DRAIN end although still on the device
    CHECK(s.cap_reached && s.outstanding == std::vector<int64_t>{1});
    for (const auto& x : f) CHECK(x.state != State::Inflight && x.state != State::Planned);
    std::printf("ok inclusive_drain_cap\n");
}

static void test_inclusive_cap_after_preemption() {
    // the issuer is preempted from before the deadline to after the cap: the timeout still fires at the cap
    Run r;
    r.cfg.request_timeout_ns = 1 * MS;
    r.cfg.drain_ns = 1 * MS;
    r.be.service = [](int64_t) { return 50 * MS; };
    r.plan(1, 10 * MS);
    r.clock.step = 1000;
    // first read after the drain begin event lands far past the cap
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    r.clock.jumps[1200] = 30 * MS;
    LoopSummary s = l.run();
    CHECK(l.facts()[0].state == State::TimedOut);
    CHECK(s.cap_reached);
    std::printf("ok inclusive_cap_after_preemption\n");
}

static void test_not_issued() {
    Run r;
    r.plan(3, 10 * MS, {1});
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    l.run();
    CHECK(states(l)[1] == State::NotIssued);
    auto ev = serialize(l);
    bool found = false;
    for (const Json& e : ev)
        if (e.at("kind").as_string() == "not_issued") {
            found = true;
            CHECK(e.at("reason").as_string() == "prepare_failed:preprocess");
            CHECK(e.at("item_indices").as_array()[0].as_int() == 1);
        }
    CHECK(found);
    CHECK(r.be.marks.size() == 1 && r.be.marks[0].second == "not_issued");
    std::printf("ok not_issued\n");
}

static void test_arrival_before_a_late_observed_completion() {
    // preemption jumps the clock past r0's device completion (~4 ms) and r1's t_sched (11 ms); r0's completion
    // is observed only after the jump (t_complete ~25 ms), so r1's arrival is earlier: it finds the slot held
    // and is dropped (earlier semantic timestamp wins; the tie order is for equal timestamps only)
    Run r;
    r.be.service = [](int64_t) { return 3 * MS; };
    r.plan(2, 10 * MS);
    r.clock.jumps[2000] = 25 * MS;  // ~1 ms after r0 was issued at ~1 ms (1 us per read)
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    l.run();
    const auto& f = l.facts();
    CHECK(f[0].state == State::Completed);
    CHECK(f[1].state == State::Dropped);
    CHECK(f[1].t_issue - f[1].t_sched > 10 * MS);  // the late admission is measured, not hidden
    CHECK(f[0].t_complete > f[1].t_sched);         // the completion was observed after r1 was due
    check_events_well_formed(serialize(l), 5);
    std::printf("ok arrival_before_a_late_observed_completion\n");
}

static void test_late_wake_times_out_before_the_observed_completion() {
    // the release-blocking case: timeout due at t_submit_begin + 1 ms, device done after 3 ms, the loop wakes
    // only at ~3.4 ms after submit -> TIMED_OUT (record time = timeout due time), then late_completion
    Run r;
    r.cfg.request_timeout_ns = 1 * MS;
    r.be.service = [](int64_t) { return 3 * MS; };
    r.plan(2, 10 * MS);
    r.clock.jumps[1100] = 4400000;  // r0: t_submit_begin 1.002 ms, done at 4.003 ms; the loop resumes at 4.4 ms
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    l.run();
    const auto& f = l.facts();
    CHECK(f[0].state == State::TimedOut && f[0].late && f[0].late_ok);
    CHECK(f[0].t_timed_out == f[0].t_submit_begin + 1 * MS);
    CHECK(f[0].t_complete >= 4400000 && f[0].t_complete > f[0].t_timed_out);
    CHECK(f[1].state == State::TimedOut);  // r1 (11 ms) found the slot free again after the late completion
    auto ev = serialize(l);
    check_events_well_formed(ev, 5);
    std::vector<std::string> kinds;
    for (const Json& e : ev)
        if (e.has("request_id") && e.at("request_id").as_int() == 0) kinds.push_back(e.at("kind").as_string());
    CHECK((kinds == std::vector<std::string>{"issued", "submit_begin", "submit_end", "timed_out", "late_completion",
                                             "device_observation"}));
    std::printf("ok late_wake_times_out_before_the_observed_completion\n");
}

static void test_completion_and_timeout_at_the_microsecond_boundary() {
    // r0: t_submit_begin = 1.002 ms, done at 1.003 ms + service; with 1 us per clock read the first poll that
    // reports completion reads t_complete = 1.003 ms + service.  timeout = service + 1 us puts the timer due
    // exactly at t_complete (tie -> COMPLETED); timeout = service puts it 1 us earlier (-> TIMED_OUT + late)
    for (int64_t extra : {int64_t(1000), int64_t(0)}) {
        Run r;
        r.be.service = [](int64_t) { return 50000; };
        r.cfg.request_timeout_ns = 50000 + extra;
        r.plan(1, 10 * MS);
        Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
        l.run();
        const RequestFacts& f0 = l.facts()[0];
        int64_t due = f0.t_submit_begin + r.cfg.request_timeout_ns;
        if (extra) {
            CHECK(f0.state == State::Completed && f0.t_complete == due);
        } else {
            CHECK(f0.state == State::TimedOut && f0.late && f0.t_timed_out == due && f0.t_complete == due + 1000);
        }
        check_events_well_formed(serialize(l), 5);
    }
    std::printf("ok completion_and_timeout_at_the_microsecond_boundary\n");
}

static void test_recorded_outcomes_follow_due_time_order_under_stalls() {
    // property test over simulated preemptions: every recorded outcome must be the one Stage 1A semantics give
    // for the recorded timestamps (completion at t_complete, timeout at t_submit_begin + timeout, arrival at
    // t_sched; ties completion -> timeout -> arrival)
    int runs = 0, completed = 0, timed_out = 0, dropped = 0, stalled_timeouts = 0;
    for (int64_t service : {int64_t(500000), 3 * MS, 7 * MS})
        for (int64_t timeout : {1 * MS, 3 * MS, 5 * MS})
            for (int64_t period : {2 * MS, 5 * MS, 10 * MS})
                for (int64_t at : {int64_t(-1), int64_t(1050), int64_t(1700), int64_t(4500), int64_t(9000)})
                    for (int64_t stall : {int64_t(1500000), int64_t(6 * MS)}) {
                        if (at < 0 && stall != 1500000) continue;
                        Run r;
                        r.cfg.request_timeout_ns = timeout;
                        r.cfg.drain_ns = 100 * MS;
                        r.be.service = [service](int64_t item) { return service + (item % 3) * 250000; };
                        r.plan(8, period);
                        Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
                        if (at >= 0) r.clock.jumps[at] = int64_t(at) * 1000 + stall;  // jump ahead from ~at us
                        LoopSummary s = l.run();
                        ++runs;
                        CHECK(!s.aborted && !s.cap_reached);
                        const auto& f = l.facts();
                        int64_t release = -1;  // when the slot held at the next arrival is released (-1: free)
                        for (const RequestFacts& x : f) {
                            bool held = release >= 0 && release > x.t_sched;  // tie: completion first
                            if (held) CHECK(x.state == State::Dropped);
                            if (!held) CHECK(x.state != State::Dropped);
                            if (x.state == State::Completed) {
                                CHECK(x.t_complete - x.t_submit_begin <= timeout);
                                release = x.t_complete;
                                ++completed;
                            } else if (x.state == State::TimedOut) {
                                CHECK(x.t_timed_out >= x.t_submit_begin + timeout);
                                CHECK(x.late && x.t_complete > x.t_submit_begin + timeout);
                                release = x.t_complete;
                                ++timed_out;
                                // a device that finished before its timer but was observed only after it
                                stalled_timeouts += x.t_submit_begin + service + 500000 < x.t_submit_begin + timeout;
                            } else if (x.state == State::Dropped) {
                                ++dropped;
                            }
                        }
                        check_events_well_formed(serialize(l), 5);
                    }
    CHECK(runs == 3 * 3 * 3 * 9);
    CHECK(completed > 0 && timed_out > 0 && dropped > 0 && stalled_timeouts > 0);  // the grid exercises every path
    std::printf("ok recorded_outcomes_follow_due_time_order_under_stalls (%d runs: %d completed, %d timed out "
                "(%d only because of a stall), %d dropped)\n", runs, completed, timed_out, stalled_timeouts, dropped);
}

static void test_asap() {
    Run r;
    r.plan(4, 0);
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    l.run();
    const auto& f = l.facts();
    for (size_t k = 0; k < f.size(); ++k) {
        CHECK(f[k].state == State::Completed);
        if (k) CHECK(f[k].t_sched >= f[k - 1].t_output_ready);
    }
    std::printf("ok asap\n");
}

static void test_errors() {
    Run r;
    r.be.submit_errors = {1};
    r.be.run_errors = {2};
    r.be.bad_output = {3};
    r.plan(5, 10 * MS);
    Loop l(r.clock, r.be, r.out, r.cfg, r.items, 0);
    l.run();
    const auto& f = l.facts();
    CHECK(f[1].state == State::Error && f[1].error_phase == "dut_submit");
    CHECK(f[2].state == State::Error && f[2].error_phase == "dut_run");
    CHECK(f[3].state == State::Error && f[3].error_phase == "postprocess");
    CHECK(f[4].state == State::Completed);  // policy continue; a submit error does not hold the slot
    auto ev = serialize(l);
    check_events_well_formed(ev, 5);
    int errors = 0;
    for (const Json& e : ev) errors += e.at("kind").as_string() == "error";
    CHECK(errors == 3);
    std::printf("ok errors\n");
}

static void test_output_bytes_match_python_encoding() {
    // expected bytes produced by Python ReferenceConvNetWorkload.encode(postprocess(raw, item 3))
    ReferenceConvNetOutput o("reference-conformance");
    int64_t logits[10] = {-5, 7, 7, 0, 1, 2, 3, 4, -9, 6};
    std::string raw(reinterpret_cast<const char*>(logits), 80), blob, phase;
    ErrorInfo err;
    CHECK(o.encode(0, 3, raw, blob, phase, err));
    CHECK(blob == "{\"item_index\":3,\"logits\":[-5,7,7,0,1,2,3,4,-9,6],\"schema\":\"sdv-ref-convnet-output/1\","
                  "\"task\":\"reference-conformance\",\"top1\":1}");
    std::printf("ok output_bytes_match_python_encoding\n");
}

static void test_canonical_json() {
    Json o = Json::object();
    o["b"] = 1;
    o["a"] = std::string("x\"y\\z\n\t\x01\xc3\xa9/");
    o["c"] = Json::array();
    o["d"] = nullptr;
    o["e"] = true;
    // Python: json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    CHECK(o.dump() == "{\"a\":\"x\\\"y\\\\z\\n\\t\\u0001\xc3\xa9/\",\"b\":1,\"c\":[],\"d\":null,\"e\":true}");
    Json p = Json::parse("{\"k\": [1, -2, \"\\u00e9\\ud83d\\ude00\", {\"z\": false}], \"j\": null}");
    CHECK(p.dump() == "{\"j\":null,\"k\":[1,-2,\"\xc3\xa9\xf0\x9f\x98\x80\",{\"z\":false}]}");
    bool threw = false;
    try {
        Json::parse("{\"a\":1,\"a\":2}");
    } catch (const JsonError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        Json::parse("{\"a\":1.5}");
    } catch (const JsonError&) {
        threw = true;
    }
    CHECK(threw);
    std::printf("ok canonical_json\n");
}

static void test_sha256() {
    CHECK(Sha256::of("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(Sha256::of("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::string m(1000000, 'a');
    CHECK(Sha256::of(m) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    Sha256 s;
    s.update(m.data(), 1);
    s.update(m.data(), 63);
    s.update(m.data(), 999936);
    CHECK(s.hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    std::printf("ok sha256\n");
}

static void test_event_writer_rejects_backwards_time() {
    EventWriter w(0, 100);
    bool threw = false;
    try {
        w.phase("x", "end", 50, "ok");
    } catch (const JsonError&) {
        threw = true;
    }
    CHECK(threw);
    std::printf("ok event_writer_rejects_backwards_time\n");
}

int main() {
    test_all_complete();
    test_drop_new();
    test_timeout_late_completion_holds_slot();
    test_inclusive_drain_cap();
    test_inclusive_cap_after_preemption();
    test_not_issued();
    test_arrival_before_a_late_observed_completion();
    test_late_wake_times_out_before_the_observed_completion();
    test_completion_and_timeout_at_the_microsecond_boundary();
    test_recorded_outcomes_follow_due_time_order_under_stalls();
    test_asap();
    test_errors();
    test_output_bytes_match_python_encoding();
    test_canonical_json();
    test_sha256();
    test_event_writer_rejects_backwards_time();
    std::printf("checks %d failures %d\n", checks, failures);
    return failures ? 1 : 0;
}
