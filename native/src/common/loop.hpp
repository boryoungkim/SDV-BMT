// SDV-BMT native runner: the realtime MEASURE/DRAIN loop (portable benchmark semantics; no vendor code).
//
// Admission semantics are those of Stage 1A (design v03 E.5), N=1, queue capacity 0, drop_new:
//   * at t_sched a request whose item failed preparation becomes NOT_ISSUED; otherwise, if the slot is
//     free it is ISSUED and submitted (INFLIGHT), else it is DROPPED(no_free_slot);
//   * the request holds the slot until its completion is observed;
//   * if t_submit_begin + request_timeout_ns passes first it becomes TIMED_OUT but keeps the slot until
//     the device work completes (submitted work cannot be cancelled); that completion is a late_completion.
// What real time changes: times are host CLOCK_MONOTONIC ns relative to the run anchor; t_issue >= t_sched
// is measured, not zero by construction; submit returns before completion and t_complete is the first poll
// that reports completion.  The issuer spins (never sleeps).
// Event order (v0.1.0): events that are already due when the loop handles them -- also after the loop woke
// late -- are decided in the order of their semantic timestamps: an arrival at t_sched, a timeout at
// t_submit_begin + request_timeout_ns, a completion at t_complete (the clock read right after the first
// poll that reports it).  The earlier timestamp wins; the Stage 1A tie order completion -> timeout ->
// arrival applies only to equal timestamps.  So a completion observed after its timeout was due is a
// TIMED_OUT request followed by a late_completion, and an arrival due before the observed completion of the
// request in the slot finds the slot held (DROPPED with drop_new).  Arrivals and timeouts already due by a
// clock read are handled before the next poll, because any completion that poll observes is later.  Record
// times: t_issue, t_complete and t_output_ready are the realized clock reads; a timed_out record carries
// the timeout's due time, or the latest earlier record time if that is later, so recorded times never
// decrease.
// DRAIN services the slot until it is free or the clock reaches cap = (end of the issue loop) + drain_ns;
// the first iteration at or after the cap is still serviced (inclusive cap, as the virtual loop processes
// events at t <= cap), so with drain_ns >= request_timeout_ns every issued request is terminal at DRAIN end.
//
// The loop records facts only (timestamps, states, device facts, output bytes and their digest) into
// preallocated memory; events are serialized after DRAIN.  It computes no metric.  The backend exposes the raw
// output as a view of its own (pinned) buffer, and the output buffers are reserved before MEASURE from the
// output pipeline's declared bound, so no raw copy and no reallocation happens in the measured path.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "sha256.hpp"

namespace sdv {

struct ErrorInfo {
    std::string code;
    std::string message;
};

struct PlannedItem {
    int64_t item_index = 0;
    int slot = -1;              // backend input slot; -1 when preparation failed
    std::string failure_stage;  // "preprocess", "prepare_input", ... when preparation failed
};

constexpr int kMaxSegments = 4;
constexpr int kMaxObservations = 8;

// Facts a backend reports for one completed request.  Keys are static strings (namespaced by the backend).
struct DeviceFacts {
    const char* timing_method = nullptr;
    int n_segments = 0;
    const char* segment_name[kMaxSegments] = {};
    int64_t segment_ns[kMaxSegments] = {};
    int n_observations = 0;
    const char* observation_key[kMaxObservations] = {};
    int64_t observation_value[kMaxObservations] = {};
    void segment(const char* name, int64_t ns) {
        segment_name[n_segments] = name;
        segment_ns[n_segments++] = ns;
    }
    void observe(const char* key, int64_t value) {
        observation_key[n_observations] = key;
        observation_value[n_observations++] = value;
    }
};

enum class State { Planned, Inflight, Completed, Error, TimedOut, Dropped, NotIssued };

struct RequestFacts {
    int64_t request_id = 0;
    int64_t item_index = 0;
    State state = State::Planned;
    int64_t t_sched = -1, t_issue = -1, t_submit_begin = -1, t_submit_end = -1;
    int64_t t_complete = -1, t_output_ready = -1, t_timed_out = -1;
    bool late = false;                  // a TIMED_OUT request whose completion was observed later
    bool late_ok = false;               // that late completion produced an output
    std::string error_phase;            // dut_submit / dut_run / postprocess / encode
    ErrorInfo error;
    DeviceFacts device;
    bool has_device = false;
    int64_t out_offset = -1, out_length = 0;
    std::string out_sha256;
};

enum class EventKind {
    MeasureBegin, MeasureEnd, MeasureFailed, DrainBegin, DrainEnd, NotIssued, Dropped, Issued, SubmitBegin,
    SubmitEnd, SubmitError, Completed, RunError, OutputError, LateCompletion, TimedOut, DeviceObservation
};

struct EventRec {
    EventKind kind;
    int64_t t;
    int64_t request_id;  // -1 for phase events
};

struct LoopConfig {
    bool asap = false;
    int64_t lead_ns = 0;
    int64_t request_timeout_ns = 0;
    int64_t drain_ns = 0;
    std::vector<int64_t> offsets_ns;  // planned offsets from t0 (fixed_rate / recorded)
    std::string replay_mode;           // "fixed_rate" / "recorded" / "asap" (recorded in the phase event)
};

struct LoopSummary {
    int64_t t_measure_begin = 0, t0 = 0, t_last_sched = 0, t_issue_end = 0, cap = 0, t_drain_end = 0;
    bool cap_reached = false;
    std::vector<int64_t> outstanding;
    bool aborted = false;
    std::string aborted_phase;
    ErrorInfo abort_error;
};

// Output pipeline: raw device output -> canonical task output bytes (postprocess + encode).
// Returns false and fills phase/err on failure.
template <class Clock, class Backend, class Output>
class RealtimeLoop {
   public:
    RealtimeLoop(Clock& clock, Backend& backend, Output& output, const LoopConfig& cfg,
                 const std::vector<PlannedItem>& items, int64_t output_base_offset)
        : clock_(clock), be_(backend), out_(output), cfg_(cfg), items_(items), output_base_(output_base_offset) {
        facts_.resize(items.size());
        events_.reserve(items.size() * 8 + 16);
        outputs_.reserve(items.size() * output.max_blob_bytes());
        blob_.reserve(std::max<size_t>(4096, output.max_blob_bytes()));
    }

    LoopSummary run() {
        LoopSummary s;
        try {
            s.t_measure_begin = clock_.now();
            s.t0 = s.t_measure_begin + cfg_.lead_ns;
            log(EventKind::MeasureBegin, s.t_measure_begin);
            s.t_last_sched = cfg_.asap ? issue_asap(s.t0) : issue_scheduled(s.t0);
            s.t_issue_end = clock_.now();
            log(EventKind::MeasureEnd, s.t_issue_end);
            phase_ = "drain";
            s.cap = s.t_issue_end + cfg_.drain_ns;
            log(EventKind::DrainBegin, clock_.now());
            while (inflight_ >= 0) {
                int64_t now = clock_.now();
                service(now, kNoArrival);
                if (now >= s.cap) break;  // the iteration at the cap was serviced
            }
            s.cap_reached = inflight_ >= 0;
            if (inflight_ >= 0) s.outstanding.push_back(facts_[size_t(inflight_)].request_id);
            s.t_drain_end = clock_.now();
            log(EventKind::DrainEnd, s.t_drain_end);
        } catch (const std::exception& e) {
            s.aborted = true;
            s.aborted_phase = phase_;
            s.abort_error = ErrorInfo{"native_exception", e.what()};
            log(EventKind::MeasureFailed, clock_.now());
        }
        summary_ = s;
        return s;
    }

    const std::vector<EventRec>& events() const { return events_; }
    const std::vector<RequestFacts>& facts() const { return facts_; }
    const std::string& outputs() const { return outputs_; }
    const LoopSummary& summary() const { return summary_; }
    const LoopConfig& config() const { return cfg_; }
    const std::string& phase() const { return phase_; }
    const std::string& item_failure(int64_t rid) const { return items_[size_t(rid)].failure_stage; }

   private:
    static constexpr int64_t kNoArrival = std::numeric_limits<int64_t>::max();

    // A completion observed by a poll but not yet decided, because an earlier-due arrival comes first.
    struct PendingCompletion {
        bool has = false;
        int status = 0;  // > 0 ok, < 0 DUT error
        ErrorInfo err;
        int64_t t = 0;  // t_complete: clock read right after the poll
    };

    Clock& clock_;
    Backend& be_;
    Output& out_;
    const LoopConfig& cfg_;
    const std::vector<PlannedItem>& items_;
    int64_t output_base_;
    std::vector<RequestFacts> facts_;
    std::vector<EventRec> events_;
    std::string outputs_;
    std::string_view raw_;
    std::string blob_;
    int64_t inflight_ = -1;  // request id holding the slot
    int64_t deadline_ = 0;
    PendingCompletion pending_;
    std::string phase_ = "measure";
    LoopSummary summary_;

    void log(EventKind k, int64_t t, int64_t rid = -1) { events_.push_back(EventRec{k, t, rid}); }

    int64_t issue_scheduled(int64_t t0) {
        int64_t t_last = t0;
        for (size_t i = 0; i < items_.size(); ++i) {
            int64_t t_sched = t0 + cfg_.offsets_ns[i];
            while (true) {
                int64_t now = clock_.now();
                if (inflight_ >= 0) service(now, t_sched);  // slot events that precede this arrival
                if (now >= t_sched) break;
            }
            admit(int64_t(i), t_sched);
            t_last = t_sched;
        }
        return t_last;
    }

    int64_t issue_asap(int64_t t0) {
        while (clock_.now() < t0) {
        }
        int64_t t_last = t0;
        for (size_t i = 0; i < items_.size(); ++i) {
            while (inflight_ >= 0) service(clock_.now(), kNoArrival);
            int64_t t_sched = clock_.now();
            admit(int64_t(i), t_sched);
            t_last = t_sched;
        }
        return t_last;
    }

    void admit(int64_t rid, int64_t t_sched) {
        int64_t now = clock_.now();
        RequestFacts& r = facts_[size_t(rid)];
        const PlannedItem& item = items_[size_t(rid)];
        r.request_id = rid;
        r.item_index = item.item_index;
        r.t_sched = t_sched;
        r.t_issue = now;  // admission decision instant (also for dropped / not-issued requests)
        if (item.slot < 0) {
            r.state = State::NotIssued;
            log(EventKind::NotIssued, now, rid);
            be_.mark_admission(rid, "not_issued");  // profiler marker hook (no-op without a profiler profile)
            return;
        }
        if (inflight_ >= 0) {
            r.state = State::Dropped;
            log(EventKind::Dropped, now, rid);
            be_.mark_admission(rid, "dropped");
            return;
        }
        r.t_submit_begin = clock_.now();
        ErrorInfo err;
        bool ok = be_.submit(rid, item.slot, err);
        r.t_submit_end = clock_.now();
        log(EventKind::Issued, now, rid);
        log(EventKind::SubmitBegin, r.t_submit_begin, rid);
        log(EventKind::SubmitEnd, r.t_submit_end, rid);
        if (!ok) {  // request-level error (policy: continue); the slot is not held
            r.state = State::Error;
            r.error_phase = "dut_submit";
            r.error = err;
            r.t_complete = r.t_submit_end;
            log(EventKind::SubmitError, clock_.now(), rid);
            return;
        }
        r.state = State::Inflight;
        inflight_ = rid;
        deadline_ = r.t_submit_begin + cfg_.request_timeout_ns;  // Stage 1A timeout timer
    }

    // Decide the events of the request in the slot that are due by `now`, in semantic-timestamp order, as far
    // as they precede the next arrival (`t_arrival`, kNoArrival when none is pending): a timeout at deadline_,
    // a completion at its t_complete.  Ties go completion -> timeout -> arrival.  Returns when nothing more
    // can be decided before the arrival or until the next clock read.
    void service(int64_t now, int64_t t_arrival) {
        while (inflight_ >= 0) {
            RequestFacts& r = facts_[size_t(inflight_)];
            const bool timer_running = r.state == State::Inflight;
            if (pending_.has) {
                if (timer_running && deadline_ < pending_.t) {  // the timeout was due before the completion
                    if (t_arrival < deadline_) return;           // ... and the arrival before the timeout
                    time_out(r);
                    continue;
                }
                if (t_arrival < pending_.t) return;  // the arrival precedes the observed completion
                PendingCompletion c = pending_;
                pending_ = PendingCompletion{};
                complete(r, c.status > 0, c.err, c.t);  // frees the slot
                return;
            }
            // any completion a poll observes now has t_complete > now: due timeouts and arrivals come first
            if (timer_running && deadline_ < now) {
                if (t_arrival < deadline_) return;
                time_out(r);
                continue;
            }
            if (t_arrival < now) return;
            ErrorInfo err;
            int status = be_.poll(err);
            if (status == 0) return;
            pending_ = PendingCompletion{true, status, err, clock_.now()};
            // loop: order the observed completion against a timeout or arrival that fell due during the poll
        }
    }

    // TIMED_OUT at the timer's due time (or the latest earlier record time, so record times never decrease).
    // The device work is not cancelled: the request keeps the slot until its (late) completion is observed.
    void time_out(RequestFacts& r) {
        r.state = State::TimedOut;
        r.t_timed_out = events_.empty() ? deadline_ : std::max(deadline_, events_.back().t);
        log(EventKind::TimedOut, r.t_timed_out, r.request_id);
    }

    void complete(RequestFacts& r, bool ok, const ErrorInfo& poll_err, int64_t t_complete) {
        bool timed_out = r.state == State::TimedOut;
        r.t_complete = t_complete;
        if (!ok) {
            r.error_phase = "dut_run";
            r.error = poll_err;
            if (timed_out) {
                r.late = true;
                log(EventKind::LateCompletion, clock_.now(), r.request_id);
            } else {
                r.state = State::Error;
                log(EventKind::RunError, clock_.now(), r.request_id);
            }
            inflight_ = -1;
            return;
        }
        raw_ = std::string_view();
        ErrorInfo collect_err;
        if (!be_.collect(r.device, raw_, collect_err)) {  // treated like an adapter exception on completion
            r.error_phase = "dut_run";
            r.error = collect_err;
            if (timed_out) {
                r.late = true;
                log(EventKind::LateCompletion, clock_.now(), r.request_id);
            } else {
                r.state = State::Error;
                log(EventKind::RunError, clock_.now(), r.request_id);
            }
            inflight_ = -1;
            return;
        }
        r.has_device = r.device.n_segments > 0 || r.device.n_observations > 0;
        blob_.clear();
        std::string phase;
        ErrorInfo out_err;
        if (!out_.encode(r.request_id, r.item_index, raw_, blob_, phase, out_err)) {
            r.error_phase = phase;
            r.error = out_err;
            int64_t t = clock_.now();
            if (timed_out) {
                r.late = true;
                log(EventKind::LateCompletion, t, r.request_id);
            } else {
                r.state = State::Error;
                log(EventKind::OutputError, t, r.request_id);
            }
        } else {
            r.out_offset = output_base_ + int64_t(outputs_.size());
            r.out_length = int64_t(blob_.size());
            r.out_sha256 = Sha256::of(blob_);
            outputs_ += blob_;  // the output record (written to outputs.bin after DRAIN)
            r.t_output_ready = clock_.now();
            if (timed_out) {
                r.late = true;
                r.late_ok = true;
                log(EventKind::LateCompletion, r.t_output_ready, r.request_id);
            } else {
                r.state = State::Completed;
                log(EventKind::Completed, r.t_output_ready, r.request_id);
            }
        }
        if (r.has_device) log(EventKind::DeviceObservation, clock_.now(), r.request_id);
        inflight_ = -1;
    }
};

}  // namespace sdv
