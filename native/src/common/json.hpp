// SDV-BMT native runner: minimal JSON value, strict reader and canonical writer (no third-party code).
//
// Reader: objects, arrays, strings (all escapes, \uXXXX incl. surrogate pairs -> UTF-8), integers
// (int64), true/false/null.  Fractions and exponents are rejected: the native runner exchanges integers
// only (time in ns, byte counts); derived floating-point values are computed by the Python side.
// Duplicate object keys are rejected.
//
// Writer: SDV-BMT canonical JSON v1, byte-identical to Python
// json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False): keys sorted by code point
// (= byte order for UTF-8), compact, strings escaped as Python does with ensure_ascii=False.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sdv {

class JsonError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

class Json {
   public:
    enum class Type { Null, Bool, Int, String, Array, Object };
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool v) : type_(Type::Bool), b_(v) {}
    Json(int v) : type_(Type::Int), i_(v) {}
    Json(long v) : type_(Type::Int), i_(v) {}
    Json(long long v) : type_(Type::Int), i_(v) {}
    Json(unsigned v) : type_(Type::Int), i_(static_cast<int64_t>(v)) {}
    Json(unsigned long v) : type_(Type::Int), i_(static_cast<int64_t>(v)) {}
    Json(unsigned long long v) : type_(Type::Int), i_(static_cast<int64_t>(v)) {}
    Json(const char* v) : type_(Type::String), s_(v) {}
    Json(std::string v) : type_(Type::String), s_(std::move(v)) {}
    Json(Array v) : type_(Type::Array), a_(std::make_shared<Array>(std::move(v))) {}
    Json(Object v) : type_(Type::Object), o_(std::make_shared<Object>(std::move(v))) {}

    static Json array() { return Json(Array{}); }
    static Json object() { return Json(Object{}); }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_string() const { return type_ == Type::String; }
    bool is_int() const { return type_ == Type::Int; }
    bool is_bool() const { return type_ == Type::Bool; }

    bool as_bool() const {
        if (type_ != Type::Bool) throw JsonError("expected a boolean");
        return b_;
    }
    int64_t as_int() const {
        if (type_ != Type::Int) throw JsonError("expected an integer");
        return i_;
    }
    const std::string& as_string() const {
        if (type_ != Type::String) throw JsonError("expected a string");
        return s_;
    }
    const Array& as_array() const {
        if (type_ != Type::Array) throw JsonError("expected an array");
        return *a_;
    }
    Array& as_array() {
        if (type_ != Type::Array) throw JsonError("expected an array");
        return *a_;
    }
    const Object& as_object() const {
        if (type_ != Type::Object) throw JsonError("expected an object");
        return *o_;
    }
    Object& as_object() {
        if (type_ != Type::Object) throw JsonError("expected an object");
        return *o_;
    }

    bool has(const std::string& key) const { return type_ == Type::Object && o_->count(key) != 0; }
    const Json& at(const std::string& key) const {
        const Object& o = as_object();
        auto it = o.find(key);
        if (it == o.end()) throw JsonError("missing key '" + key + "'");
        return it->second;
    }
    Json& operator[](const std::string& key) { return as_object()[key]; }
    void push(Json v) { as_array().push_back(std::move(v)); }

    std::string dump() const {
        std::string out;
        write(out);
        return out;
    }
    void write(std::string& out) const;

    static Json parse(const std::string& text);

   private:
    Type type_ = Type::Null;
    bool b_ = false;
    int64_t i_ = 0;
    std::string s_;
    std::shared_ptr<Array> a_;
    std::shared_ptr<Object> o_;
};

inline void write_json_string(std::string& out, const std::string& s) {
    static const char* hex = "0123456789abcdef";
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 0xF]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

inline void Json::write(std::string& out) const {
    switch (type_) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += b_ ? "true" : "false"; break;
        case Type::Int: out += std::to_string(i_); break;
        case Type::String: write_json_string(out, s_); break;
        case Type::Array: {
            out.push_back('[');
            bool first = true;
            for (const Json& v : *a_) {
                if (!first) out.push_back(',');
                first = false;
                v.write(out);
            }
            out.push_back(']');
            break;
        }
        case Type::Object: {
            out.push_back('{');
            bool first = true;
            for (const auto& kv : *o_) {  // std::map: byte order = code point order for UTF-8 keys
                if (!first) out.push_back(',');
                first = false;
                write_json_string(out, kv.first);
                out.push_back(':');
                kv.second.write(out);
            }
            out.push_back('}');
            break;
        }
    }
}

namespace detail {

class Reader {
   public:
    explicit Reader(const std::string& t) : t_(t) {}

    Json document() {
        Json v = value();
        ws();
        if (p_ != t_.size()) fail("trailing characters");
        return v;
    }

   private:
    const std::string& t_;
    size_t p_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw JsonError("JSON " + what + " at offset " + std::to_string(p_));
    }
    void ws() {
        while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\r' || t_[p_] == '\t')) ++p_;
    }
    bool lit(const char* s) {
        size_t n = std::char_traits<char>::length(s);
        if (t_.compare(p_, n, s) == 0) {
            p_ += n;
            return true;
        }
        return false;
    }
    Json value() {
        ws();
        if (p_ >= t_.size()) fail("unexpected end");
        char c = t_[p_];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') return Json(string());
        if (lit("true")) return Json(true);
        if (lit("false")) return Json(false);
        if (lit("null")) return Json(nullptr);
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        fail("unexpected character");
    }
    Json number() {
        size_t start = p_;
        if (t_[p_] == '-') ++p_;
        if (p_ >= t_.size() || t_[p_] < '0' || t_[p_] > '9') fail("bad number");
        if (t_[p_] == '0' && p_ + 1 < t_.size() && t_[p_ + 1] >= '0' && t_[p_ + 1] <= '9') fail("leading zero");
        while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        if (p_ < t_.size() && (t_[p_] == '.' || t_[p_] == 'e' || t_[p_] == 'E'))
            fail("non-integer number (the native runner exchanges integers only)");
        try {
            return Json(static_cast<long long>(std::stoll(t_.substr(start, p_ - start))));
        } catch (const std::out_of_range&) {
            fail("integer out of int64 range");
        }
    }
    static void utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    uint32_t hex4() {
        if (p_ + 4 > t_.size()) fail("short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = t_[p_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }
    std::string string() {
        ++p_;  // opening quote
        std::string out;
        while (true) {
            if (p_ >= t_.size()) fail("unterminated string");
            unsigned char c = static_cast<unsigned char>(t_[p_++]);
            if (c == '"') return out;
            if (c < 0x20) fail("control character in string");
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (p_ >= t_.size()) fail("unterminated escape");
            char e = t_[p_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (!(p_ + 2 <= t_.size() && t_[p_] == '\\' && t_[p_ + 1] == 'u')) fail("lone surrogate");
                        p_ += 2;
                        uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("lone surrogate");
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    Json array() {
        ++p_;
        Json out = Json::array();
        ws();
        if (p_ < t_.size() && t_[p_] == ']') {
            ++p_;
            return out;
        }
        while (true) {
            out.push(value());
            ws();
            if (p_ >= t_.size()) fail("unterminated array");
            char c = t_[p_++];
            if (c == ']') return out;
            if (c != ',') fail("expected ',' or ']'");
        }
    }
    Json object() {
        ++p_;
        Json out = Json::object();
        ws();
        if (p_ < t_.size() && t_[p_] == '}') {
            ++p_;
            return out;
        }
        while (true) {
            ws();
            if (p_ >= t_.size() || t_[p_] != '"') fail("expected a key");
            std::string key = string();
            if (out.has(key)) fail("duplicate key '" + key + "'");
            ws();
            if (p_ >= t_.size() || t_[p_++] != ':') fail("expected ':'");
            out[key] = value();
            ws();
            if (p_ >= t_.size()) fail("unterminated object");
            char c = t_[p_++];
            if (c == '}') return out;
            if (c != ',') fail("expected ',' or '}'");
        }
    }
};

}  // namespace detail

inline Json Json::parse(const std::string& text) { return detail::Reader(text).document(); }

}  // namespace sdv
