// my_recorder.cpp
//
// Records a RealSense D400 stream to data/<dir>/<name>.db3.


#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace pt = boost::property_tree;
namespace po = boost::program_options;
using namespace std::chrono_literals;

static_assert(std::atomic<bool>::is_always_lock_free, "atomic<bool> must be lock-free for signal use");
static std::atomic<bool> g_running{true};
// First Ctrl+C: ask the capture loop to stop gracefully (recording gets finalised).
// Second Ctrl+C: the user really wants out (e.g. something is stuck) -> exit immediately.
static void signal_handler(int)
{
    if (!g_running.exchange(false))
        _exit(130);
}

// ============================================================================
// Small utilities
// ============================================================================

static int64_t unix_ms_now()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static std::string utc_now_iso()
{
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

static std::string json_escape(const std::string& s)
{
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
                else          o << c;
        }
    }
    return o.str();
}

static std::string csv_quote(const std::string& s)
{
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    return out + "\"";
}

static std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static bool is_true(const std::string& s)
{
    const std::string l = lower(s);
    return l == "true" || l == "1" || l == "on";
}

static std::string fnv1a64_hex(const std::string& data)
{
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : data) { h ^= c; h *= 1099511628211ULL; }
    std::ostringstream o;
    o << std::hex << std::setw(16) << std::setfill('0') << h;
    return o.str();
}

// write to a temp file, then rename -> a crash can never leave a half-written file
static void write_file_atomic(const fs::path& path, const std::string& content)
{
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("Cannot write " + tmp.string());
        f << content;
        f.flush();
        if (!f) throw std::runtime_error("Write failed for " + tmp.string());
    }
    fs::rename(tmp, path);
}

// Runs fn() in a helper thread
template <class F>
static void run_with_timeout(const std::string& what, std::chrono::seconds timeout,
                             const std::string& hint, F&& fn)
{
    auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(fn));
    auto fut  = task->get_future();
    std::thread([task] { (*task)(); }).detach();
    if (fut.wait_for(timeout) != std::future_status::ready) {
        std::cerr << "\nERROR: " << what << " did not finish within " << timeout.count() << " s.\n" << hint << std::endl;
        std::_Exit(4);
    }
    fut.get();
}



struct ConfigFile {
    std::string path;
    std::string text;
    pt::ptree   tree;
    std::string hash;   // FNV-1a of the file contents -> proves which settings were used
};

static ConfigFile read_config_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open config file '" + path + "' (current directory: " +
                                 fs::current_path().string() + "). Use --config <path>.");
    ConfigFile c;
    c.path = path;
    c.text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (c.text.empty())
        throw std::runtime_error("Config file '" + path + "' is empty.");
    try {
        std::istringstream is(c.text);
        pt::read_json(is, c.tree);
    } catch (const pt::json_parser_error& e) {
        throw std::runtime_error("Config file '" + path + "' is not valid JSON: " + e.what());
    }
    c.hash = fnv1a64_hex(c.text);
    return c;
}

static std::optional<std::string> cfg_get(const pt::ptree& t, const std::string& path)
{
    auto v = t.get_optional<std::string>(pt::ptree::path_type(path, '/'));
    if (v) return *v;
    return std::nullopt;
}

struct StreamRequest {
    int         width  = 640;
    int         height = 480;
    int         fps    = 15;
    std::string depth_format = "Z16";
};

// Checks that need no camera: does the JSON agree with what we are about to request?
static void validate_config_vs_request(const ConfigFile& cfg, const StreamRequest& req,
                                       std::vector<std::string>& errors,
                                       std::vector<std::string>& warnings)
{
    const pt::ptree& t = cfg.tree;

    if (auto sv = cfg_get(t, "schema version"); sv && *sv != "1")
        warnings.push_back("config 'schema version' is " + *sv + " (expected 1)");

    if (!t.get_child_optional("parameters"))
        errors.push_back("config has no \"parameters\" section - load_json would have nothing to apply");

    // "viewer" holds stream settings that ONLY the RealSense Viewer reads; load_json ignores
    // them. So it is on us to make sure our own enable_stream() calls agree with them.
    auto check_int = [&](const char* key, int requested, const char* what) {
        auto v = cfg_get(t, std::string("viewer/") + key);
        if (!v) { warnings.push_back(std::string("config has no viewer/") + key + " (cannot cross-check " + what + ")"); return; }
        int parsed = 0;
        try { parsed = std::stoi(*v); }
        catch (...) { errors.push_back(std::string("viewer/") + key + " = '" + *v + "' is not a number"); return; }
        if (parsed != requested)
            errors.push_back(std::string(what) + ": program requests " + std::to_string(requested) +
                             " but config viewer/" + key + " = " + *v);
    };
    check_int("stream-width",  req.width,  "stream width");
    check_int("stream-height", req.height, "stream height");
    check_int("stream-fps",    req.fps,    "stream fps");

    if (auto v = cfg_get(t, "viewer/stream-depth-format")) {
        if (lower(*v) != lower(req.depth_format))
            errors.push_back("depth format: program requests " + req.depth_format +
                             " but config viewer/stream-depth-format = " + *v);
    } else {
        warnings.push_back("config has no viewer/stream-depth-format (cannot cross-check depth format)");
    }

    // Manual depth exposure (microseconds) can't be longer than one frame period.
    auto ae  = cfg_get(t, "parameters/controls-autoexposure-auto");
    auto exp = cfg_get(t, "parameters/controls-autoexposure-manual");
    if (ae && exp && !is_true(*ae)) {
        try {
            const double exposure_us = std::stod(*exp);
            const double period_us   = 1e6 / req.fps;
            if (exposure_us >= period_us)
                errors.push_back("manual depth exposure " + *exp + " us is >= the frame period at " +
                                 std::to_string(req.fps) + " fps (" + std::to_string(int(period_us)) +
                                 " us): the camera could not keep this frame rate");
        } catch (...) {
            errors.push_back("parameters/controls-autoexposure-manual = '" + *exp + "' is not a number");
        }
    }
}


// Device handling

static rs2::device wait_for_device(rs2::context& ctx, const std::string& serial,
                                   bool need_advanced_enabled, std::chrono::seconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline && g_running) {
        try {
            auto list = ctx.query_devices();
            for (auto&& d : list) {
                if (!d.supports(RS2_CAMERA_INFO_SERIAL_NUMBER) ||
                    serial != d.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER))
                    continue;
                if (!need_advanced_enabled) return d;
                if (d.is<rs400::advanced_mode>() && d.as<rs400::advanced_mode>().is_enabled()) return d;
            }
        } catch (const rs2::error&) {
            // device is mid-reset; keep polling
        }
        std::this_thread::sleep_for(500ms);
    }
    throw std::runtime_error("Timed out waiting for device " + serial + " to come back after advanced-mode reset.");
}

static std::optional<rs2::sensor> find_sensor(const rs2::device& dev, const std::string& name)
{
    for (auto&& s : dev.query_sensors())
        if (s.supports(RS2_CAMERA_INFO_NAME) && name == s.get_info(RS2_CAMERA_INFO_NAME))
            return s;
    return std::nullopt;
}

static bool device_supports_profile(const rs2::device& dev, rs2_stream stream, rs2_format fmt,
                                    int w, int h, int fps)
{
    for (auto&& s : dev.query_sensors())
        for (auto&& p : s.get_stream_profiles()) {
            if (p.stream_type() != stream || p.format() != fmt || p.fps() != fps) continue;
            if (!p.is<rs2::video_stream_profile>()) continue;
            auto vp = p.as<rs2::video_stream_profile>();
            if (vp.width() == w && vp.height() == h) return true;
        }
    return false;
}

// Opens the first device. If a config file was given: makes sure it matches the camera,
// enables advanced mode (waiting for the reset properly) and loads the JSON.
// Without a config nothing is changed on the camera. Must run BEFORE the pipeline starts.
static rs2::device open_and_configure_device(rs2::context& ctx, const ConfigFile* cfg,
                                             std::vector<std::string>& warnings,
                                             bool load_json, std::chrono::seconds load_timeout)
{
    auto list = ctx.query_devices();
    if (list.size() == 0)
        throw std::runtime_error("No RealSense device detected.");

    rs2::device dev = list[0];
    const std::string serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    const std::string name   = dev.get_info(RS2_CAMERA_INFO_NAME);
    std::cout << "Device: " << name << " (serial " << serial << ")\n";
    if (dev.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)) {
        const std::string usb = dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR);
        std::cout << "USB link: " << usb << "\n";
        if (!usb.empty() && usb[0] == '2')
            warnings.push_back("camera is on a USB " + usb + " link (not USB 3). Firmware commands such as "
                               "load_json are slow and fragile there: try a blue USB 3 port and a short, good cable");
    }

    if (!cfg) {
        std::cout << "No --config given: recording with the camera's current settings (no JSON is loaded).\n";
        return dev;
    }

    if (auto jn = cfg_get(cfg->tree, "device/name"); jn && *jn != name)
        throw std::runtime_error("Config was exported for '" + *jn + "' but the connected camera is '" +
                                 name + "'. Refusing to load it.");
    if (dev.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION))
        if (auto jf = cfg_get(cfg->tree, "device/fw version");
            jf && *jf != dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION))
            warnings.push_back("config was exported with firmware " + *jf + " but the camera runs " +
                               dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION));

    if (!load_json) {
        warnings.push_back("JSON NOT loaded (--no-load-json): the camera keeps whatever settings it currently has");
        return dev;
    }

    if (!dev.is<rs400::advanced_mode>())
        throw std::runtime_error("This device does not support advanced mode (needed for JSON loading).");

    bool toggled = false;
    {
        auto adv = dev.as<rs400::advanced_mode>();
        if (!adv.is_enabled()) {
            std::cout << "Enabling advanced mode (the camera will reset, this takes a few seconds)...\n";
            adv.toggle_advanced_mode(true);
            toggled = true;
        }
    }
    if (toggled) {
        // Old handles are dead after the reset: never reuse them, fetch a fresh device.
        std::this_thread::sleep_for(2s);
        dev = wait_for_device(ctx, serial, /*need_advanced_enabled=*/true, 30s);
        std::cout << "Advanced mode enabled.\n";
    }

    auto adv = dev.as<rs400::advanced_mode>();
    std::cout << "Loading camera settings from " << cfg->path << " (hash " << cfg->hash << ")...\n";
    const auto t0 = std::chrono::steady_clock::now();
    try {
        run_with_timeout("load_json", load_timeout,
            "The camera's firmware is not answering. Usual causes: another program has the camera open\n"
            "(RealSense Viewer, or an earlier hung my_recorder: check `pgrep -a my_recorder`), a USB 2.x link\n"
            "or bad cable/port, or a JSON this firmware chokes on. Unplug and replug the camera, then check\n"
            "that the same JSON loads in the RealSense Viewer. Use --no-load-json to test the rest of the setup.",
            [adv, text = cfg->text]() mutable { adv.load_json(text); });
    } catch (const rs2::error& e) {
        throw std::runtime_error(std::string("load_json failed in ") + e.get_failed_function() + ": " + e.what());
    }
    std::this_thread::sleep_for(300ms);
    std::cout << "Settings loaded in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s.\n";
    return dev;
}

// Read a few options back from the camera and compare with the JSON, so a silently
// ignored setting doesn't go unnoticed. Returns the list of mismatches.
struct SettingCheck {
    const char* json_key;
    const char* label;
    const char* sensor;
    rs2_option  option;
    enum Kind { Number, Bool, OnOff } kind;
    const char* only_if_false_key;   // skip when this JSON key is true (e.g. exposure while AE is on)
};

static std::vector<std::string> verify_applied_settings(const rs2::device& dev, const pt::ptree& tree)
{
    static const SettingCheck checks[] = {
        {"controls-laserpower",                "laser power",               "Stereo Module", RS2_OPTION_LASER_POWER,              SettingCheck::Number, nullptr},
        {"controls-laserstate",                "emitter",                   "Stereo Module", RS2_OPTION_EMITTER_ENABLED,          SettingCheck::OnOff,  nullptr},
        {"controls-autoexposure-auto",         "depth auto-exposure",       "Stereo Module", RS2_OPTION_ENABLE_AUTO_EXPOSURE,     SettingCheck::Bool,   nullptr},
        {"controls-autoexposure-manual",       "depth exposure (us)",       "Stereo Module", RS2_OPTION_EXPOSURE,                 SettingCheck::Number, "controls-autoexposure-auto"},
        {"controls-depth-gain",                "depth gain",                "Stereo Module", RS2_OPTION_GAIN,                     SettingCheck::Number, nullptr},
        {"controls-color-autoexposure-auto",   "color auto-exposure",       "RGB Camera",    RS2_OPTION_ENABLE_AUTO_EXPOSURE,     SettingCheck::Bool,   nullptr},
        {"controls-color-white-balance-auto",  "color auto white balance",  "RGB Camera",    RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE,SettingCheck::Bool,   nullptr},
    };

    std::vector<std::string> mismatches;
    std::cout << "Read-back of applied settings:\n";
    for (const auto& c : checks) {
        auto jv = cfg_get(tree, std::string("parameters/") + c.json_key);
        if (!jv) continue;
        if (c.only_if_false_key)
            if (auto gate = cfg_get(tree, std::string("parameters/") + c.only_if_false_key); gate && is_true(*gate))
                continue;

        double expected = 0;
        try {
            expected = (c.kind == SettingCheck::Number) ? std::stod(*jv) : (is_true(*jv) ? 1.0 : 0.0);
        } catch (...) { continue; }

        auto sensor = find_sensor(dev, c.sensor);
        if (!sensor || !sensor->supports(c.option)) continue;

        const double actual = sensor->get_option(c.option);
        const bool ok = std::abs(actual - expected) <= std::max(1.0, 0.01 * std::abs(expected))
                        || (c.kind != SettingCheck::Number && (actual != 0) == (expected != 0));
        std::cout << "  " << std::left << std::setw(26) << c.label << " json=" << std::setw(8) << *jv
                  << " camera=" << std::setw(8) << actual << (ok ? "OK" : "MISMATCH") << "\n";
        if (!ok)
            mismatches.push_back(std::string(c.label) + ": json says " + *jv + ", camera reports " + std::to_string(actual));
    }
    return mismatches;
}


// TTL trigger

struct FrameInfo {
    int64_t     index = -1;               // 0-based index of the frameset as received by this program
    uint64_t    color_frame_number = 0;   // SDK/hardware frame counters: robust ids, use these when
    uint64_t    depth_frame_number = 0;   //   matching against the recording at playback time
    double      timestamp_ms = 0;
    std::string timestamp_domain;
    int64_t     host_unix_ms = 0;         // host clock when this frameset arrived
};

struct TriggerInfo {
    bool        valid = false;
    std::string source;                   // "ttl" or "software_warmup"
    int         byte = -1;
    int64_t     host_unix_ms = 0;
    FrameInfo   last_frame;               // last frameset received BEFORE the trigger
};

struct SharedState {
    std::mutex            m;
    FrameInfo             latest;
    TriggerInfo           trigger;        // first trigger only
    std::atomic<bool>     triggered{false};
    std::atomic<int64_t>  trigger_after_index{-1};
    std::atomic<int>      trigger_count{0};
    std::atomic<bool>     stop_listener{false};
    std::atomic<bool>     listener_failed{false};
    std::string           listener_error;
};

static void register_trigger(SharedState& st, const std::string& source, int byte)
{
    const int64_t now = unix_ms_now();
    std::lock_guard<std::mutex> lk(st.m);
    st.trigger_count++;
    if (st.trigger.valid) return;
    st.trigger.valid        = true;
    st.trigger.source       = source;
    st.trigger.byte         = byte;
    st.trigger.host_unix_ms = now;
    st.trigger.last_frame   = st.latest;
    st.trigger_after_index  = st.latest.index;
    st.triggered            = true;
}

static int open_trigger_port(const std::string& port, speed_t baud = B9600)
{
    int fd = open(port.c_str(), O_RDONLY | O_NOCTTY | O_SYNC);
    if (fd < 0)
        throw std::runtime_error("Could not open trigger port " + port + " (" + std::strerror(errno) + ")");

    auto fail = [&](const std::string& msg) { close(fd); throw std::runtime_error(msg + " on " + port); };

    struct termios tty{};
    if (tcgetattr(fd, &tty) != 0) fail("tcgetattr failed");
    if (cfsetispeed(&tty, baud) != 0 || cfsetospeed(&tty, baud) != 0) fail("setting baud rate failed");

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | ICRNL | INLCR | PARMRK | INPCK | ISTRIP);
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) fail("tcsetattr failed");
    tcflush(fd, TCIFLUSH);  
    return fd;
}

// Runs in its own thread so trigger timestamps are not quantised to the frame period.
// select() wakes up as soon as a byte arrives; the timeout is only to notice shutdown.
static void ttl_listener(int fd, SharedState& st)
{
    while (g_running && !st.stop_listener) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        timeval tv{0, 50'000};
        int r = select(fd + 1, &fds, nullptr, nullptr, &tv);
        if (r < 0) {
            if (errno == EINTR) continue;
            st.listener_error = std::string("select() failed: ") + std::strerror(errno);
            st.listener_failed = true;
            return;
        }
        if (r == 0) continue;

        uint8_t buf[64];
        ssize_t n = read(fd, buf, sizeof buf);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            st.listener_error = std::string("read() failed: ") + std::strerror(errno);
            st.listener_failed = true;
            return;
        }
        for (ssize_t i = 0; i < n; i++)
            if (buf[i] != 0x00)                   // 0x00 = falling edge, not a trigger
                register_trigger(st, "ttl", buf[i]);
    }
}


// Output files

struct RunSummary {
    std::string video_file;
    std::string device_name, device_serial, device_fw;
    StreamRequest req;
    const ConfigFile* cfg = nullptr;
    int         length_s = 0;
    int64_t     frames_total = 0;
    int64_t     dropped_frames = 0;
    bool        completed = false;
    bool        config_applied = true;
    std::string stop_reason;
    TriggerInfo trigger;
    int         trigger_count = 0;
};

static std::string frame_json(const FrameInfo& f, const std::string& ind)
{
    std::ostringstream o;
    o << "{\n"
      << ind << "  \"index\": " << f.index << ",\n"
      << ind << "  \"color_frame_number\": " << f.color_frame_number << ",\n"
      << ind << "  \"depth_frame_number\": " << f.depth_frame_number << ",\n"
      << ind << "  \"timestamp_ms\": " << std::fixed << std::setprecision(3) << f.timestamp_ms << ",\n"
      << ind << "  \"timestamp_domain\": \"" << json_escape(f.timestamp_domain) << "\",\n"
      << ind << "  \"host_unix_ms\": " << f.host_unix_ms << "\n"
      << ind << "}";
    return o.str();
}

static std::string summary_json(const RunSummary& s)
{
    std::ostringstream o;
    o << "{\n"
      << "  \"schema\": 1,\n"
      << "  \"file\": \"" << json_escape(s.video_file) << "\",\n"
      << "  \"created_utc\": \"" << utc_now_iso() << "\",\n"
      << "  \"librealsense\": \"" << RS2_API_VERSION_STR << "\",\n"
      << "  \"device\": {\"name\": \"" << json_escape(s.device_name) << "\", \"serial\": \""
                       << json_escape(s.device_serial) << "\", \"firmware\": \"" << json_escape(s.device_fw) << "\"},\n"
      << "  \"stream\": {\"width\": " << s.req.width << ", \"height\": " << s.req.height
                       << ", \"fps\": " << s.req.fps << ", \"depth_format\": \"" << s.req.depth_format
                       << "\", \"color_format\": \"RGB8\"},\n"
      << "  \"config\": ";
    if (s.cfg)
        o << "{\"path\": \"" << json_escape(s.cfg->path) << "\", \"fnv1a64\": \"" << s.cfg->hash
          << "\", \"applied\": " << (s.config_applied ? "true" : "false") << "},\n";
    else
        o << "null,\n";

    if (s.trigger.valid) {
        o << "  \"trigger\": {\n"
          << "    \"source\": \"" << s.trigger.source << "\",\n"
          << "    \"byte\": " << s.trigger.byte << ",\n"
          << "    \"host_unix_ms\": " << s.trigger.host_unix_ms << ",\n"
          << "    \"trigger_count_seen\": " << s.trigger_count << ",\n"
          << "    \"first_frame_index_after_trigger\": " << (s.trigger.last_frame.index + 1) << ",\n"
          << "    \"last_frame_before_trigger\": " << frame_json(s.trigger.last_frame, "    ") << "\n"
          << "  },\n";
    } else {
        o << "  \"trigger\": null,\n";
    }

    o << "  \"summary\": {\n"
      << "    \"length_after_trigger_s\": " << s.length_s << ",\n"
      << "    \"frames_total\": " << s.frames_total << ",\n"
      << "    \"dropped_frames_detected\": " << s.dropped_frames << ",\n"
      << "    \"completed\": " << (s.completed ? "true" : "false") << ",\n"
      << "    \"stop_reason\": \"" << json_escape(s.stop_reason) << "\"\n"
      << "  }\n"
      << "}\n";
    return o.str();
}

static void append_csv_row(const fs::path& csv, const RunSummary& s, const std::string& name)
{
    const bool exists = fs::exists(csv);
    std::ofstream f(csv, std::ios::app);
    if (!f) throw std::runtime_error("Cannot append to " + csv.string());
    if (!exists)
        f << "name,trigger_source,first_frame_index_after_trigger,last_color_frame_number_before_trigger,"
             "last_depth_frame_number_before_trigger,trigger_unix_ms,fps,frames_total,dropped_frames,completed,config_fnv1a64\n";
    f << csv_quote(name) << ',';
    if (s.trigger.valid)
        f << s.trigger.source << ',' << (s.trigger.last_frame.index + 1) << ','
          << s.trigger.last_frame.color_frame_number << ',' << s.trigger.last_frame.depth_frame_number << ','
          << s.trigger.host_unix_ms << ',';
    else
        f << "none,,,,,";
    f << s.req.fps << ',' << s.frames_total << ',' << s.dropped_frames << ','
      << (s.completed ? "true" : "false") << ',' << (s.cfg ? s.cfg->hash : std::string()) << '\n';
}


// RAII helpers so the recording is always finalised and the thread/port released

struct PipelineGuard {
    rs2::pipeline& pipe;
    bool active = false;
    explicit PipelineGuard(rs2::pipeline& p) : pipe(p) {}
    void stop() { if (active) { active = false; pipe.stop(); } }
    ~PipelineGuard() { if (active) { try { pipe.stop(); } catch (...) {} } }
};

struct FdGuard {
    int fd = -1;
    ~FdGuard() { if (fd >= 0) close(fd); }
};

struct ListenerGuard {
    SharedState& st;
    std::thread  t;
    explicit ListenerGuard(SharedState& s) : st(s) {}
    ~ListenerGuard() { st.stop_listener = true; if (t.joinable()) t.join(); }
};


static int run(int argc, char* argv[])
{
    std::string name, dir, trigger_port, config_path;
    int    fps = 15, length = 300;
    double warmup_s = 2.0, ttl_timeout_s = 0.0;
    int    load_timeout_s = 45;
    bool   overwrite = false, strict = false, no_load_json = false;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show help message")
        ("fps,f",    po::value<int>(&fps)->required(),
            "frames per second (must agree with viewer/stream-fps in the config)")
        ("name,e",   po::value<std::string>(&name)->required(),
            "recording name -> data/<dir>/<name>.db3")
        ("dir,d",    po::value<std::string>(&dir)->default_value(""),
            "sub-directory inside data/")
        ("length,l", po::value<int>(&length)->default_value(300),
            "seconds to keep recording AFTER the trigger")
        ("trigger-port,t", po::value<std::string>(&trigger_port)->default_value(""),
            "serial port of the TTL trigger (e.g. /dev/ttyUSB0). Empty: a software marker is placed "
            "after --warmup seconds instead")
        ("warmup",   po::value<double>(&warmup_s)->default_value(2.0),
            "warm-up seconds before the software marker (only used without --trigger-port)")
        ("ttl-timeout", po::value<double>(&ttl_timeout_s)->default_value(0.0),
            "abort if no trigger arrives within this many seconds (0 = wait forever)")
        ("config,c", po::value<std::string>(&config_path)->default_value(""),
            "OPTIONAL camera settings JSON from the RealSense Viewer. If omitted, no JSON is loaded and the "
            "camera records 640x480 Z16 + RGB8 at --fps with its current settings")
        ("load-timeout", po::value<int>(&load_timeout_s)->default_value(45),
            "give up if loading the JSON takes longer than this many seconds")
        ("no-load-json", po::bool_switch(&no_load_json),
            "do NOT load the JSON (diagnostics only; settings are still read back and reported)")
        ("strict",   po::bool_switch(&strict),
            "treat camera-settings read-back mismatches as errors")
        ("overwrite", po::bool_switch(&overwrite),
            "allow overwriting an existing recording");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << '\n'; return 0; }
    po::notify(vm);

    if (fps <= 0 || length <= 0 || warmup_s < 0 || ttl_timeout_s < 0 || load_timeout_s <= 0)
        throw std::runtime_error("--fps, --length and --load-timeout must be > 0; --warmup and --ttl-timeout must be >= 0");

    const StreamRequest req{640, 480, fps, "Z16"};

    // ---- 1. Pre-flight checks that need no camera ---------------------------
    fs::path out_dir    = fs::path("data") / dir;
    fs::path video_file = out_dir / (name + ".db3");
    fs::path sidecar    = out_dir / (name + ".trigger.json");
    if (fs::exists(video_file) && !overwrite)
        throw std::runtime_error(video_file.string() + " already exists (use --overwrite to replace it).");

    std::vector<std::string> errors, warnings;
    std::optional<ConfigFile> cfg;
    if (!config_path.empty()) {
        cfg = read_config_file(config_path);
        validate_config_vs_request(*cfg, req, errors, warnings);
        for (auto& w : warnings) std::cerr << "WARNING: " << w << '\n';
        if (!errors.empty()) {
            std::cerr << "\nConfiguration is inconsistent, NOT starting the camera:\n";
            for (auto& e : errors) std::cerr << "  - " << e << '\n';
            return 3;
        }
    } else if (no_load_json) {
        std::cerr << "WARNING: --no-load-json has no effect without --config\n";
    }

    // Open the trigger port now: if it is unavailable we want to find out before recording.
    FdGuard trigger_fd;
    if (!trigger_port.empty())
        trigger_fd.fd = open_trigger_port(trigger_port);

    // ---- 2. Camera: configure, verify, check stream support -----------------
    rs2::context ctx;
    std::vector<std::string> dev_warnings;
    rs2::device  dev = open_and_configure_device(ctx, cfg ? &*cfg : nullptr, dev_warnings,
                                                 !no_load_json, std::chrono::seconds(load_timeout_s));
    for (auto& w : dev_warnings) std::cerr << "WARNING: " << w << '\n';

    if (cfg) {
        auto mismatches = verify_applied_settings(dev, cfg->tree);
        if (!mismatches.empty()) {
            for (auto& m : mismatches) std::cerr << (strict ? "ERROR: " : "WARNING: ") << "setting not applied - " << m << '\n';
            if (strict) return 3;
        }
    }

    if (!device_supports_profile(dev, RS2_STREAM_DEPTH, RS2_FORMAT_Z16,  req.width, req.height, req.fps) ||
        !device_supports_profile(dev, RS2_STREAM_COLOR, RS2_FORMAT_RGB8, req.width, req.height, req.fps)) {
        std::cerr << "ERROR: the camera does not offer " << req.width << "x" << req.height << " @ " << req.fps
                  << " fps for Z16 depth + RGB8 color. NOT starting.\n";
        return 3;
    }

    RunSummary sum;
    sum.device_name   = dev.get_info(RS2_CAMERA_INFO_NAME);
    sum.device_serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    sum.device_fw     = dev.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION) ? dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) : "";
    sum.req = req;
    sum.cfg = cfg ? &*cfg : nullptr;
    sum.length_s = length;
    sum.config_applied = cfg && !no_load_json;
    sum.video_file = video_file.string();

    // ---- 3. Start pipeline + recording --------------------------------------
    fs::create_directories(out_dir);

    rs2::pipeline pipe(ctx);
    rs2::config   rs_cfg;
    rs_cfg.enable_device(sum.device_serial);
    rs_cfg.enable_stream(RS2_STREAM_DEPTH, req.width, req.height, RS2_FORMAT_Z16,  req.fps);
    rs_cfg.enable_stream(RS2_STREAM_COLOR, req.width, req.height, RS2_FORMAT_RGB8, req.fps);
    rs_cfg.enable_record_to_file(video_file.string());

    // From here on Ctrl+C means "stop gracefully and finalise the file" (2nd Ctrl+C = force quit).
    // During the setup above it keeps its default behaviour, so it can never leave you stuck.
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    PipelineGuard pipe_guard(pipe);
    pipe.start(rs_cfg);
    pipe_guard.active = true;
    std::cout << "Recording to " << video_file << " (started at frame 0).\n";

    SharedState st;
    ListenerGuard listener(st);
    const bool use_ttl = trigger_fd.fd >= 0;
    if (use_ttl) {
        listener.t = std::thread(ttl_listener, trigger_fd.fd, std::ref(st));
        std::cout << "Waiting for TTL trigger on " << trigger_port << " ... (recording continues meanwhile)\n";
    } else {
        std::cout << "No trigger port: software marker after " << warmup_s << " s of warm-up.\n";
    }

    // ---- 4. Capture loop -----------------------------------------------------
    const int64_t warmup_frames = static_cast<int64_t>(std::llround(warmup_s * fps));
    const int64_t frames_after  = static_cast<int64_t>(length) * fps;
    const auto    t_start       = std::chrono::steady_clock::now();

    int64_t  idx = -1, dropped = 0;
    uint64_t prev_color_no = 0;
    int      consecutive_timeouts = 0;
    std::string stop_reason = "interrupted";

    while (g_running) {
        rs2::frameset fsr;
        try {
            fsr = pipe.wait_for_frames(5000);
            consecutive_timeouts = 0;
        } catch (const rs2::error& e) {
            std::cerr << "WARNING: wait_for_frames failed: " << e.what() << '\n';
            if (++consecutive_timeouts >= 3) { stop_reason = "camera stopped delivering frames"; break; }
            continue;
        }

        idx++;
        auto color = fsr.get_color_frame();
        auto depth = fsr.get_depth_frame();

        FrameInfo fi;
        fi.index        = idx;
        fi.host_unix_ms = unix_ms_now();
        if (color) {
            fi.color_frame_number = color.get_frame_number();
            fi.timestamp_ms       = color.get_timestamp();
            fi.timestamp_domain   = rs2_timestamp_domain_to_string(color.get_frame_timestamp_domain());
            if (prev_color_no != 0 && fi.color_frame_number > prev_color_no + 1) {
                dropped += static_cast<int64_t>(fi.color_frame_number - prev_color_no - 1);
                std::cerr << "WARNING: color frames " << prev_color_no + 1 << ".." << fi.color_frame_number - 1
                          << " missing (index " << idx << ")\n";
            }
            prev_color_no = fi.color_frame_number;
        }
        if (depth) fi.depth_frame_number = depth.get_frame_number();
        {
            std::lock_guard<std::mutex> lk(st.m);
            st.latest = fi;
        }

        // software marker when no TTL hardware is used
        if (!use_ttl && !st.triggered && idx + 1 >= warmup_frames)
            register_trigger(st, "software_warmup", -1);

        if (st.listener_failed) { stop_reason = "trigger port error: " + st.listener_error; break; }

        const bool trig = st.triggered;
        if (trig && idx - st.trigger_after_index >= frames_after) { stop_reason = "completed"; break; }

        if (!trig && use_ttl && ttl_timeout_s > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() > ttl_timeout_s) {
            stop_reason = "no TTL trigger within --ttl-timeout";
            break;
        }

        if ((idx + 1) % fps == 0) {
            if (trig)
                std::cout << "\rframe " << idx << "  | trigger at frame " << st.trigger_after_index.load()
                          << "  | +" << (idx - st.trigger_after_index) << "/" << frames_after << "   " << std::flush;
            else
                std::cout << "\rframe " << idx << "  | waiting for trigger...   " << std::flush;
        }
    }
    std::cout << '\n';

    // ---- 5. Finalise ---------------------------------------------------------
    listener.st.stop_listener = true;
    pipe_guard.stop();   // flushes and closes the .db3

    {
        std::lock_guard<std::mutex> lk(st.m);
        sum.trigger = st.trigger;
    }
    sum.trigger_count  = st.trigger_count;
    sum.frames_total   = idx + 1;
    sum.dropped_frames = dropped;
    sum.completed      = (stop_reason == "completed");
    sum.stop_reason    = stop_reason;

    write_file_atomic(sidecar, summary_json(sum));
    append_csv_row(out_dir / "triggers.csv", sum, name);

    std::cout << "Stopped: " << stop_reason << "\n"
              << "Frames recorded: " << sum.frames_total << " (dropped detected: " << dropped << ")\n";
    if (sum.trigger.valid)
        std::cout << "Trigger (" << sum.trigger.source << "): first frame after trigger = index "
                  << sum.trigger.last_frame.index + 1 << " (last color frame number before it: "
                  << sum.trigger.last_frame.color_frame_number << ")\n";
    else
        std::cout << "No trigger was received.\n";
    std::cout << "Wrote " << sidecar << " and " << (out_dir / "triggers.csv") << '\n';

    return sum.completed ? 0 : 2;
}

int main(int argc, char* argv[])
{
    try {
        return run(argc, argv);
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error in " << e.get_failed_function() << "(" << e.get_failed_args() << "): "
                  << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }
    return EXIT_FAILURE;
}