// SDV-BMT native runner: serialization of recorded facts as raw events (events.jsonl, schema version 2).
//
// The field set of every event kind is the one the bundle reader validates (src/sdv_bmt/observation/
// events.py).  Lines are canonical JSON; common fields are schema_version, kind, seq, t_run_ns and
// run_clock_domain.  Serialization happens after DRAIN, in the order the facts were recorded.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "json.hpp"
#include "loop.hpp"

namespace sdv {

constexpr int64_t kEventSchemaVersion = 2;
const char* const kRunClockDomain = "monotonic";
const char* const kRequestPhase = "measure";

inline Json error_json(const ErrorInfo& e) {
    Json o = Json::object();
    o["code"] = e.code;
    o["message"] = e.message;
    return o;
}

class EventWriter {
   public:
    EventWriter(int64_t next_seq, int64_t last_t) : seq_(next_seq), last_t_(last_t) {}

    // Appends one canonical JSON line; enforces non-decreasing run-clock time.
    void add(Json fields, const std::string& kind, int64_t t_run_ns) {
        if (t_run_ns < last_t_) throw JsonError("run clock went backwards in native events");
        fields["schema_version"] = kEventSchemaVersion;
        fields["kind"] = kind;
        fields["seq"] = seq_++;
        fields["t_run_ns"] = t_run_ns;
        fields["run_clock_domain"] = kRunClockDomain;
        fields.write(text_);
        text_.push_back('\n');
        last_t_ = t_run_ns;
    }
    void phase(const std::string& name, const std::string& edge, int64_t t, const std::string& outcome = "",
               const Json* details = nullptr, const ErrorInfo* error = nullptr) {
        Json f = Json::object();
        f["phase"] = name;
        f["edge"] = edge;
        if (!outcome.empty()) f["outcome"] = outcome;
        if (details) f["details"] = *details;
        if (error) f["error"] = error_json(*error);
        add(std::move(f), "phase", t);
    }

    int64_t next_seq() const { return seq_; }
    int64_t last_t() const { return last_t_; }
    const std::string& text() const { return text_; }

   private:
    int64_t seq_;
    int64_t last_t_;
    std::string text_;
};

struct EventContext {
    std::string codec;          // output codec recorded in output_ref, e.g. "canonical-json-utf8/1"
    std::string outputs_file = "outputs.bin";
};

inline Json output_ref_json(const RequestFacts& r, const EventContext& ctx) {
    Json o = Json::object();
    o["request_id"] = r.request_id;
    o["file"] = ctx.outputs_file;
    o["offset"] = r.out_offset;
    o["length"] = r.out_length;
    o["sha256"] = r.out_sha256;
    o["codec"] = ctx.codec;
    return o;
}

// Writes the loop's recorded events (MEASURE/DRAIN) in recording order.
template <class Loop>
void write_loop_events(const Loop& loop, EventWriter& w, const EventContext& ctx) {
    const LoopSummary& s = loop.summary();
    const LoopConfig& cfg = loop.config();
    const auto& facts = loop.facts();
    for (const EventRec& e : loop.events()) {
        const RequestFacts* r = e.request_id >= 0 ? &facts[size_t(e.request_id)] : nullptr;
        Json f = Json::object();
        auto admission = [&](Json& o) {
            o["request_id"] = r->request_id;
            Json items = Json::array();
            items.push(r->item_index);
            o["item_indices"] = items;
            o["phase"] = kRequestPhase;
            o["t_sched"] = r->t_sched;
        };
        switch (e.kind) {
            case EventKind::MeasureBegin: {
                Json d = Json::object();
                d["planned_requests"] = int64_t(facts.size());
                d["replay_mode"] = cfg.replay_mode;
                d["t0_ns"] = s.t0;
                d["wait_policy"] = "spin";
                w.phase("measure", "begin", e.t, "", &d);
                break;
            }
            case EventKind::MeasureEnd: w.phase("measure", "end", e.t, "ok"); break;
            case EventKind::DrainBegin: {
                Json d = Json::object();
                d["drain_ns"] = cfg.drain_ns;
                d["cap_t_run_ns"] = s.cap;
                w.phase("drain", "begin", e.t, "", &d);
                break;
            }
            case EventKind::DrainEnd: {
                Json d = Json::object();
                d["cap_reached"] = s.cap_reached;
                Json ids = Json::array();
                for (int64_t id : s.outstanding) ids.push(id);
                d["outstanding_request_ids"] = ids;
                w.phase("drain", "end", e.t, "ok", &d);
                break;
            }
            case EventKind::MeasureFailed:
                w.phase(s.aborted_phase, "end", e.t, "failed", nullptr, &s.abort_error);
                break;
            case EventKind::NotIssued:
                admission(f);
                f["reason"] = "prepare_failed:" + loop.item_failure(r->request_id);
                w.add(std::move(f), "not_issued", e.t);
                break;
            case EventKind::Dropped:
                admission(f);
                f["reason"] = "no_free_slot";
                w.add(std::move(f), "dropped", e.t);
                break;
            case EventKind::Issued:
                admission(f);
                f["t_issue"] = r->t_issue;
                w.add(std::move(f), "issued", e.t);
                break;
            case EventKind::SubmitBegin:
                f["request_id"] = r->request_id;
                w.add(std::move(f), "submit_begin", e.t);
                break;
            case EventKind::SubmitEnd:
                f["request_id"] = r->request_id;
                w.add(std::move(f), "submit_end", e.t);
                break;
            case EventKind::SubmitError:
            case EventKind::RunError:
            case EventKind::OutputError:
                f["request_id"] = r->request_id;
                f["error_phase"] = r->error_phase;
                f["error"] = error_json(r->error);
                f["t_complete"] = r->t_complete;
                w.add(std::move(f), "error", e.t);
                break;
            case EventKind::Completed:
                f["request_id"] = r->request_id;
                f["t_complete"] = r->t_complete;
                f["t_output_ready"] = r->t_output_ready;
                f["output_ref"] = output_ref_json(*r, ctx);
                w.add(std::move(f), "completed", e.t);
                break;
            case EventKind::LateCompletion:
                f["request_id"] = r->request_id;
                f["t_complete"] = r->t_complete;
                if (r->late_ok) {
                    f["result"] = "ok";
                    f["t_output_ready"] = r->t_output_ready;
                    f["output_ref"] = output_ref_json(*r, ctx);
                } else {
                    f["result"] = "error";
                    f["error_phase"] = r->error_phase;
                    f["error"] = error_json(r->error);
                }
                w.add(std::move(f), "late_completion", e.t);
                break;
            case EventKind::TimedOut:
                f["request_id"] = r->request_id;
                f["timeout_ns"] = cfg.request_timeout_ns;
                w.add(std::move(f), "timed_out", e.t);
                break;
            case EventKind::DeviceObservation: {
                f["request_id"] = r->request_id;
                const DeviceFacts& d = r->device;
                if (d.n_segments) {
                    Json timing = Json::object();
                    for (int k = 0; k < d.n_segments; ++k) timing[d.segment_name[k]] = d.segment_ns[k];
                    f["timing"] = timing;
                    f["timing_method"] = d.timing_method ? d.timing_method : "";
                } else {
                    f["timing"] = nullptr;
                    f["timing_method"] = nullptr;
                }
                Json obs = Json::object();
                for (int k = 0; k < d.n_observations; ++k) obs[d.observation_key[k]] = d.observation_value[k];
                f["observations"] = obs;
                w.add(std::move(f), "device_observation", e.t);
                break;
            }
        }
    }
}

}  // namespace sdv
