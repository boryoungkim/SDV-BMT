// SDV-BMT native runner: phase protocol between the Python control plane and the native measurement plane.
//
//   sdv-bmt-native-<backend> describe '<json args>'   -> one JSON line (backend identity), exit
//   sdv-bmt-native-<backend> run SPEC.json             -> phase protocol on stdin/stdout
//
// Protocol: the runner pins itself to spec.cpu_affinity and answers {"reply":"ready",...}; then each stdin
// line is one command and gets exactly one stdout line in reply ({"reply": <command>, "ok": bool, ...}).
// Commands arrive only at lifecycle phase boundaries, never per request:
//   prepare  -> DUT.prepare (backend: device, program, persistent buffers)
//   stage    -> stage every prepared input from the staging file, re-verifying its SHA-256; an item may carry
//               an "output_context" object for the workload's native output pipeline (e.g. original image size)
//   measure  -> precondition, backend snapshot, MEASURE/DRAIN, backend snapshot; appends the recorded events
//               to events.jsonl and the outputs to outputs.bin (the Python recorder is suspended meanwhile and
//               hands over seq / last run-clock time / output offset)
//   cleanup  -> DUT.cleanup, then exit
// The runner records facts; it derives no metric.  The request counts in the measure reply are a
// cross-check for the Python side, which derives the canonical counts from the events.
#pragma once

#include <fcntl.h>
#include <sched.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "events.hpp"
#include "json.hpp"
#include "loop.hpp"
#include "output_pipeline.hpp"

namespace sdv {

struct MonotonicClock {
    int64_t anchor = 0;
    int64_t now() const {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec - anchor;
    }
};

inline void reply(const Json& j) {
    std::string line = j.dump();
    line.push_back('\n');
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);
}

inline Json fail_reply(const std::string& command, const std::string& code, const std::string& message) {
    Json r = Json::object();
    r["reply"] = command;
    r["ok"] = false;
    r["error"] = error_json(ErrorInfo{code, message});
    return r;
}

inline void append_file(const std::string& path, const std::string& data, int64_t expect_size) {
    int fd = ::open(path.c_str(), O_WRONLY | O_APPEND);
    if (fd < 0) throw JsonError("cannot open " + path + " for append");
    if (expect_size >= 0) {
        struct stat st;
        if (fstat(fd, &st) != 0 || int64_t(st.st_size) != expect_size) {
            ::close(fd);
            throw JsonError(path + " does not have the handed-over size");
        }
    }
    size_t done = 0;
    while (done < data.size()) {
        ssize_t n = ::write(fd, data.data() + done, data.size() - done);
        if (n <= 0) {
            ::close(fd);
            throw JsonError("write failed: " + path);
        }
        done += size_t(n);
    }
    ::close(fd);
}

inline std::vector<int> realized_affinity() {
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<int> out;
    if (sched_getaffinity(0, sizeof set, &set) == 0)
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &set)) out.push_back(c);
    return out;
}

template <class Backend>
class NativeRunner {
   public:
    explicit NativeRunner(const Json& spec)
        : spec_(spec), be_(spec), output_(spec.at("output_pipeline")) {
        if (spec.at("schema").as_string() != "sdv-bmt-native-run/1") throw JsonError("unsupported spec schema");
        clock_.anchor = spec.at("anchor_monotonic_ns").as_int();
        const Json& sched = spec.at("schedule");
        cfg_.replay_mode = sched.at("mode").as_string();
        cfg_.asap = cfg_.replay_mode == "asap";
        cfg_.lead_ns = sched.at("lead_ns").as_int();
        if (!cfg_.asap)
            for (const Json& v : sched.at("offsets_ns").as_array()) cfg_.offsets_ns.push_back(v.as_int());
        const Json& run = spec.at("run");
        if (run.at("max_inflight").as_int() != 1 || run.at("queue_capacity").as_int() != 0 ||
            run.at("on_full").as_string() != "drop_new")
            throw JsonError("the native runner implements N=1, queue 0, drop_new only");
        cfg_.request_timeout_ns = run.at("request_timeout_ns").as_int();
        cfg_.drain_ns = run.at("drain_ns").as_int();
        precondition_ = spec.at("precondition_requests").as_int();
        ctx_.codec = spec.at("output_pipeline").at("codec").as_string();
        cpu_set_t set;
        CPU_ZERO(&set);
        for (const Json& c : spec.at("cpu_affinity").as_array()) CPU_SET(int(c.as_int()), &set);
        sched_setaffinity(0, sizeof set, &set);  // the realized affinity is reported, never assumed
    }

    int serve() {
        Json ready = Json::object();
        ready["reply"] = "ready";
        ready["ok"] = true;
        ready["pid"] = int64_t(getpid());
        Json aff = Json::array();
        for (int c : realized_affinity()) aff.push(c);
        ready["realized_affinity"] = aff;
        reply(ready);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::string command = "unknown";
            try {
                Json cmd = Json::parse(line);
                command = cmd.at("command").as_string();
                if (command == "prepare") {
                    Json r = Json::object();
                    r["reply"] = command;
                    r["ok"] = true;
                    r["facts"] = be_.prepare(spec_);
                    reply(r);
                } else if (command == "stage") {
                    reply(stage(cmd));
                } else if (command == "measure") {
                    reply(measure(cmd));
                } else if (command == "cleanup") {
                    Json r = Json::object();
                    r["reply"] = command;
                    r["facts"] = be_.cleanup();
                    r["ok"] = !r["facts"].has("busy") || !r["facts"].at("busy").as_bool();
                    reply(r);
                    return 0;
                } else {
                    reply(fail_reply(command, "unknown_command", "unknown command"));
                }
            } catch (const std::exception& e) {
                reply(fail_reply(command, "native_exception", e.what()));
            }
        }
        be_.cleanup();  // stdin closed without cleanup: release the device anyway
        return 0;
    }

   private:
    Json spec_;
    Backend be_;
    OutputPipeline output_;
    MonotonicClock clock_;
    LoopConfig cfg_;
    EventContext ctx_;
    int64_t precondition_ = 0;
    std::vector<PlannedItem> items_;

    Json stage(const Json& cmd) {
        const std::string& path = cmd.at("path").as_string();
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0 && !cmd.at("items").as_array().empty()) throw JsonError("cannot open staging file");
        Json results = Json::array();
        items_.clear();
        int slot = 0;
        std::string buf;
        for (const Json& it : cmd.at("items").as_array()) {
            PlannedItem p;
            p.item_index = it.at("item_index").as_int();
            if (it.has("output_context")) {
                output_.set_context(p.item_index, it.at("output_context"));
            } else if (output_.needs_context() && !it.has("failure_stage")) {
                throw JsonError("output pipeline " + output_.kind() + " needs an output_context for every item");
            }
            Json res = Json::object();
            res["item_index"] = p.item_index;
            if (it.has("failure_stage")) {
                p.failure_stage = it.at("failure_stage").as_string();
                res["ok"] = false;
                res["stage"] = p.failure_stage;
            } else {
                size_t n = size_t(it.at("nbytes").as_int());
                buf.resize(n);
                ssize_t got = ::pread(fd, buf.data(), n, off_t(it.at("offset").as_int()));
                ErrorInfo err;
                bool ok = got == ssize_t(n) &&
                          be_.stage(slot, p.item_index, buf.data(), n, it.at("sha256").as_string(), err);
                if (got != ssize_t(n)) err = ErrorInfo{"StagingReadError", "short read from the staging file"};
                if (ok) {
                    p.slot = slot++;
                    res["ok"] = true;
                } else {
                    p.failure_stage = "prepare_input";
                    res["ok"] = false;
                    res["stage"] = "prepare_input";
                    res["error"] = error_json(err);
                }
            }
            items_.push_back(p);
            results.push(res);
        }
        if (fd >= 0) ::close(fd);
        if (!cfg_.asap && cfg_.offsets_ns.size() != items_.size())
            throw JsonError("planned offsets and staged items differ in number");
        Json r = Json::object();
        r["reply"] = "stage";
        r["ok"] = true;
        r["items"] = results;
        r["staged"] = int64_t(slot);
        return r;
    }

    template <class Loop>
    static Json counts(const Loop& loop) {
        int64_t planned = 0, issued = 0, completed = 0, error = 0, timed_out = 0, dropped = 0, not_issued = 0,
                late = 0, unresolved = 0;
        for (const RequestFacts& r : loop.facts()) {
            ++planned;
            switch (r.state) {
                case State::Completed: ++issued; ++completed; break;
                case State::Error: ++issued; ++error; break;
                case State::TimedOut: ++issued; ++timed_out; break;
                case State::Inflight: ++issued; ++unresolved; break;
                case State::Dropped: ++dropped; break;
                case State::NotIssued: ++not_issued; break;
                case State::Planned: ++unresolved; break;
            }
            late += r.late;
        }
        Json c = Json::object();
        c["planned"] = planned;
        c["issued"] = issued;
        c["completed"] = completed;
        c["error"] = error;
        c["timed_out"] = timed_out;
        c["dropped"] = dropped;
        c["not_issued"] = not_issued;
        c["late_completion"] = late;
        c["unresolved"] = unresolved;
        return c;
    }

    bool precondition(Json& summary, ErrorInfo& err) {
        std::vector<int> slots;
        for (const PlannedItem& p : items_)
            if (p.slot >= 0) slots.push_back(p.slot);
        int64_t completed = 0, errors = 0;
        summary = Json::object();
        summary["requested"] = precondition_;
        if (precondition_ && slots.empty()) {
            summary["completed"] = int64_t(0);
            summary["errors"] = int64_t(0);
            summary["note"] = "no prepared item";
            return true;
        }
        for (int64_t k = 0; k < precondition_; ++k) {
            ErrorInfo e;
            if (!be_.submit(-(k + 1), slots[size_t(k) % slots.size()], e)) {
                err = ErrorInfo{"precondition_submit_failed", e.code + ": " + e.message};
                return false;
            }
            int64_t deadline = clock_.now() + cfg_.request_timeout_ns;
            while (true) {
                int st = be_.poll(e);
                if (st != 0) {
                    completed += st > 0;
                    errors += st < 0;
                    break;
                }
                if (clock_.now() >= deadline) {
                    err = ErrorInfo{"precondition_timeout", "precondition request " + std::to_string(k) +
                                                                " did not complete within the request timeout"};
                    return false;
                }
            }
        }
        summary["completed"] = completed;
        summary["errors"] = errors;
        return true;
    }

    void observe(EventWriter& w, const std::string& point) {
        std::string name = "dut.observe." + point;
        try {
            Json snap = be_.observe(point);
            w.phase(name, "end", clock_.now(), "ok", &snap);
        } catch (const std::exception& e) {
            ErrorInfo err{"observe_failed", e.what()};
            w.phase(name, "end", clock_.now(), "failed", nullptr, &err);
        }
    }

    Json measure(const Json& cmd) {
        EventWriter w(cmd.at("next_seq").as_int(), cmd.at("last_t_run_ns").as_int());
        int64_t output_offset = cmd.at("output_offset").as_int();
        const std::string& events_path = spec_.at("files").at("events").as_string();
        const std::string& outputs_path = spec_.at("files").at("outputs").as_string();
        Json r = Json::object();
        r["reply"] = "measure";
        if (precondition_) {
            Json d = Json::object();
            d["requests"] = precondition_;
            w.phase("precondition", "begin", clock_.now(), "", &d);
            Json summary;
            ErrorInfo err;
            if (!precondition(summary, err)) {
                w.phase("precondition", "end", clock_.now(), "failed", nullptr, &err);
                append_file(events_path, w.text(), -1);
                r["ok"] = false;
                r["failed_step"] = "precondition";
                r["error"] = error_json(err);
                r["next_seq"] = w.next_seq();
                r["last_t_run_ns"] = w.last_t();
                r["output_offset"] = output_offset;
                return r;
            }
            w.phase("precondition", "end", clock_.now(), "ok", &summary);
            r["precondition"] = summary;
        }
        observe(w, "measure_begin");
        RealtimeLoop<MonotonicClock, Backend, OutputPipeline> loop(clock_, be_, output_, cfg_, items_, output_offset);
        LoopSummary s = loop.run();
        write_loop_events(loop, w, ctx_);
        if (!s.aborted) observe(w, "drain_end");
        append_file(outputs_path, loop.outputs(), output_offset);
        append_file(events_path, w.text(), -1);
        Json summ = Json::object();
        summ["t_measure_begin"] = s.t_measure_begin;
        summ["t0"] = s.t0;
        summ["t_last_sched"] = s.t_last_sched;
        summ["t_issue_end"] = s.t_issue_end;
        summ["cap"] = s.cap;
        summ["t_drain_end"] = s.t_drain_end;
        summ["cap_reached"] = s.cap_reached;
        Json ids = Json::array();
        for (int64_t id : s.outstanding) ids.push(id);
        summ["outstanding_request_ids"] = ids;
        r["ok"] = !s.aborted;
        if (s.aborted) {
            r["failed_step"] = "measure";
            r["error"] = error_json(s.abort_error);
        }
        r["summary"] = summ;
        r["native_counts"] = counts(loop);
        r["next_seq"] = w.next_seq();
        r["last_t_run_ns"] = w.last_t();
        r["output_offset"] = output_offset + int64_t(loop.outputs().size());
        return r;
    }
};

template <class Backend>
int native_main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::string(argv[1]) == "describe") {
            reply(Backend::describe(Json::parse(argv[2])));
            return 0;
        }
        if (argc >= 3 && std::string(argv[1]) == "run") {
            std::FILE* f = std::fopen(argv[2], "rb");
            if (!f) throw JsonError("cannot read spec");
            std::string text;
            char buf[65536];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
            std::fclose(f);
            NativeRunner<Backend> runner(Json::parse(text));
            return runner.serve();
        }
        if (argc >= 2 && std::string(argv[1]) == "encode-reference") {
            // test helper: stdin lines {"item_index", "task", "logits": [10 ints]} -> {"item_index", "blob"}
            std::string line;
            while (std::getline(std::cin, line)) {
                Json in = Json::parse(line);
                ReferenceConvNetOutput out(in.at("task").as_string());
                std::string raw, blob, phase;
                for (const Json& v : in.at("logits").as_array()) {
                    int64_t x = v.as_int();
                    raw.append(reinterpret_cast<const char*>(&x), 8);
                }
                ErrorInfo err;
                Json o = Json::object();
                o["item_index"] = in.at("item_index").as_int();
                if (out.encode(0, in.at("item_index").as_int(), raw, blob, phase, err)) o["blob"] = blob;
                else o["error"] = error_json(err);
                reply(o);
            }
            return 0;
        }
        if (argc >= 3 && std::string(argv[1]) == "encode-output") {
            // test helper: argv[2] = output_pipeline spec JSON; stdin lines {"item_index", "raw_path", "raw_offset",
            // "raw_bytes", "output_context"?} -> {"item_index", "blob"} or {"item_index", "error", "phase"}
            OutputPipeline out(Json::parse(argv[2]));
            std::string line;
            while (std::getline(std::cin, line)) {
                Json in = Json::parse(line);
                const int64_t item = in.at("item_index").as_int();
                if (in.has("output_context")) out.set_context(item, in.at("output_context"));
                std::FILE* f = std::fopen(in.at("raw_path").as_string().c_str(), "rb");
                if (!f) throw JsonError("cannot read raw_path");
                std::string raw(size_t(in.at("raw_bytes").as_int()), '\0');
                std::fseek(f, long(in.at("raw_offset").as_int()), SEEK_SET);
                size_t got = std::fread(raw.data(), 1, raw.size(), f);
                std::fclose(f);
                if (got != raw.size()) throw JsonError("short raw read");
                std::string blob, phase;
                ErrorInfo err;
                Json o = Json::object();
                o["item_index"] = item;
                if (out.encode(0, item, raw, blob, phase, err)) {
                    o["blob"] = blob;
                } else {
                    o["error"] = error_json(err);
                    o["phase"] = phase;
                }
                reply(o);
            }
            return 0;
        }
        std::fprintf(stderr, "usage: %s describe JSON | run SPEC.json | encode-reference | encode-output SPEC\n",
                     argv[0]);
        return 2;
    } catch (const std::exception& e) {
        Json r = fail_reply("startup", "native_exception", e.what());
        reply(r);
        return 2;
    }
}

}  // namespace sdv
