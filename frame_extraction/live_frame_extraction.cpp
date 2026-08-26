#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <csignal>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <filesystem>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

static bool running = true;
void signal_handler(int signum) { running = false; }


// Async PLY save queue
struct SaveJob {
    rs2::points points;
    rs2::video_frame color;
    int  index;
    std::string result_path;
    rs2::frame filtered;
    int total_frames;
    int cloud_index;
};


std::queue<SaveJob>  save_queue;
std::mutex           queue_mutex;

// thread safe boolean: If 2 threads use variable at same time, it forces each step to execute its interrupted transacrtion
std::atomic<bool>    save_thread_running{true};


void save_points_roi(
    const rs2::points&      points,
    const rs2::video_frame& color,
    int                     index,
    const std::string&      result_path)
{
    std::ostringstream filename;
    filename << result_path << "/pointcloud_"
             << std::setw(5) << std::setfill('0') << index << ".ply";

    auto vertices   = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();
    int w = color.get_width();
    int h = color.get_height();
    auto color_data = reinterpret_cast<const uint8_t*>(color.get_data());

    struct Vertex3D { float x, y, z; };
    std::vector<std::vector<Vertex3D>> verts_2d(h, std::vector<Vertex3D>(w));
    std::vector<std::vector<std::pair<float,float>>> tex_2d(
        h, std::vector<std::pair<float,float>>(w));

    for (int row = 0; row < h; row++)
        for (int col = 0; col < w; col++) {
            int i = row * w + col;
            verts_2d[row][col] = { vertices[i].x, vertices[i].y, vertices[i].z };
            tex_2d[row][col]   = { tex_coords[i].u, tex_coords[i].v };
        }

    // ROI — hardcoded here, or pass as parameter if needed
    int x0 = 400, x1 = 920, y0 = 25, y1 = 525;

    size_t valid = 0;
    for (int row = y0; row < y1; row++)
        for (int col = x0; col < x1; col++)
            if (verts_2d[row][col].z > 0) valid++;

    if (valid == 0) {
        std::cout << "Frame " << index << ": empty ROI, skipping\n";
        return;
    }

    std::ofstream ofs(filename.str());
    ofs << "ply\nformat ascii 1.0\n"
        << "element vertex " << valid << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "end_header\n";

    for (int row = y0; row < y1; row++)
        for (int col = x0; col < x1; col++) {
            const auto& v = verts_2d[row][col];
            if (v.z <= 0) continue;
            const auto& tc = tex_2d[row][col];
            int cx = std::min(std::max(int(tc.first  * w), 0), w - 1);
            int cy = std::min(std::max(int(tc.second * h), 0), h - 1);
            int px = (cy * w + cx) * 3;
            ofs << v.x << " " << v.y << " " << v.z << " "
                << int(color_data[px + 2]) << " "
                << int(color_data[px + 1]) << " "
                << int(color_data[px + 0]) << "\n";
        }

    std::cout << "Saved " << filename.str() << " (" << valid << " pts)\n";
}


void preview_live(int cloud_index, int total_frames, rs2::frame filtered, rs2::video_frame color)
{
    // Color preview
    cv::Mat color_mat(
        color.get_height(), color.get_width(),
        CV_8UC3,
        (void*)color.get_data()
    );
    // Draw ROI rectangle on preview
    cv::rectangle(color_mat,
        cv::Point(400, 25), cv::Point(920, 525),
        cv::Scalar(0, 255, 0), 2);
    // Draw frame counter
    cv::putText(color_mat,
        "Frame " + std::to_string(cloud_index) + "/" + std::to_string(total_frames),
        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 255, 0), 2);
    cv::putText(color_mat,
        "Queue: " + std::to_string(save_queue.size()),
        cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(0, 200, 255), 2);

    rs2::colorizer colorize;
    rs2::video_frame depth_colored = colorize.colorize(filtered);
    cv::Mat depth_mat(
        depth_colored.get_height(), depth_colored.get_width(),
        CV_8UC3,
        (void*)depth_colored.get_data()
    );
    cv::rectangle(depth_mat,
        cv::Point(400, 25), cv::Point(920, 525),
        cv::Scalar(0, 255, 0), 2);

    // ── Show side by side ─────────────────────────────────────────────
    cv::Mat preview;
    cv::hconcat(color_mat, depth_mat, preview);
    cv::resize(preview, preview, cv::Size(), 0.6, 0.6);  // shrink to fit screen
    cv::imshow("RealSense Preview (press Q to stop)", preview);

    int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') running = false;
}


void save_worker()
{
    while (save_thread_running || !save_queue.empty())
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (save_queue.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Save next job (captured frame) with no concurrency
            SaveJob job = std::move(save_queue.front());
            save_queue.pop();
            save_points_roi(job.points, job.color, job.index, job.result_path);
            preview_live(job.cloud_index, job.total_frames, job.filtered, job.color);
        }
    }
}


int main(int argc, char* argv[]) try
{
    std::signal(SIGINT, signal_handler);
    namespace po = boost::program_options;

    std::string name, dir = "", trigger_port = "";
    int fps = 0, length = 20;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h",         "show help message")
        ("fps,f",          po::value<int>(&fps)->required(),         "capture fps")
        ("name,e",         po::value<std::string>(&name)->required(), "experiment name")
        ("length,l",       po::value<int>(&length),                  "recording length (s)")
        ("dir,d",          po::value<std::string>(&dir),             "sub-directory")
        ("trigger-port,t", po::value<std::string>(&trigger_port),    "TTL serial port");
 
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << '\n'; return 0; }
    po::notify(vm);

    std::string result_path = "results/" + (dir.empty() ? "" : dir + "/") + name;
    std::filesystem::create_directories(result_path);
    std::cout << "Output: " << result_path << '\n';

    // 1. Camera settings

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0)
        throw std::runtime_error("No RealSense device detected.");

    rs2::device dev = devices[0];

    // Enable advanced mode for json
    if (!dev.is<rs400::advanced_mode>())
        throw std::runtime_error("Device does not support advanced mode.");

    auto advanced_mode = dev.as<rs400::advanced_mode>();

    if (!advanced_mode.is_enabled())
    {
        std::cout << "Enabling advanced mode...\n";
        advanced_mode.toggle_advanced_mode(true);

        std::this_thread::sleep_for(std::chrono::seconds(3));
        devices = ctx.query_devices();
        dev = devices[0];
        advanced_mode = dev.as<rs400::advanced_mode>();
    }

    // std::cout << "Loading camera settings from JSON...\n";
    // std::string json_content = load_camera_config("../camera_settings/threshold_settings.json");
    // advanced_mode.load_json(json_content);
    // std::cout << "Settings loaded.\n";


    // 2. Recording setup with parallel thread for PLY points capturing
    rs2::pipeline pipe;
    rs2::config   cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16,  fps);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, fps);
    cfg.enable_record_to_file(name + ".db3");   // db3 recorded in parallel

    // Filters
    rs2::threshold_filter    thr_filter;
    rs2::disparity_transform depth_to_disparity(true);
    rs2::spatial_filter      spat_filter;
    rs2::temporal_filter     temp_filter;
    rs2::disparity_transform disparity_to_depth(false);
    rs2::hole_filling_filter hole_filter;

    thr_filter.set_option(RS2_OPTION_MIN_DISTANCE,        0.1f);
    thr_filter.set_option(RS2_OPTION_MAX_DISTANCE,        1.0f);
    spat_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE,   2);
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA,0.5f);
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA,20);
    spat_filter.set_option(RS2_OPTION_HOLES_FILL,         0);
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA,0.4f);
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA,20);
    temp_filter.set_option(RS2_OPTION_HOLES_FILL,         0);

    pipe.start(cfg);
    auto t_start = std::chrono::steady_clock::now();


    rs2::pointcloud pc;
    rs2::align      align_to_color(RS2_STREAM_COLOR);

    // Start async PLY save thread
    std::thread saver(save_worker);

    // Camera delivers frames at hardware rate. We only save every (1/fps) seconds (ex: 1/6 = every 100 ms)
    const auto frame_interval = std::chrono::microseconds(1'000'000 / fps);
    auto next_save_time       = std::chrono::steady_clock::now();

    int cloud_index  = 0;
    int total_frames = length * fps;

    std::cout << "Recording " << length << "s at " << fps
              << " fps → " << total_frames << " frames\n";

    // auto total_filter_time = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
    auto total_filter_time = 0;

    while (running && cloud_index < total_frames)
    {
        auto t_capture = std::chrono::steady_clock::now();

        rs2::frameset frames;

        // If no frames
        if (!pipe.poll_for_frames(&frames)) continue;

        auto aligned = align_to_color.process(frames);
        auto color   = aligned.get_color_frame();
        auto depth   = aligned.get_depth_frame();
        if (!color || !depth) continue;

        rs2::frame filtered = depth;
        filtered = thr_filter.process(filtered);
        filtered = depth_to_disparity.process(filtered);
        filtered = spat_filter.process(filtered);
        filtered = temp_filter.process(filtered);
        filtered = disparity_to_depth.process(filtered);
        filtered = hole_filter.process(filtered);

        auto t_filtered = std::chrono::steady_clock::now();
        auto filter_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                              t_filtered - t_capture).count();

        // PLY save
        // Only enqueue a save job when we've passed the next scheduled save time.
        // This ensures PLY output matches the requested fps exactly.
        auto now = std::chrono::steady_clock::now();
        if (now >= next_save_time)
        {
            cloud_index++;
            next_save_time += frame_interval; 

            pc.map_to(color);
            rs2::points pts = pc.calculate(filtered);  

            // Push to async queue with captured thread return
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                save_queue.push({ pts, color, cloud_index, result_path, filtered, total_frames });
            }

            auto t_enqueued = std::chrono::steady_clock::now();
            auto enqueue_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  t_enqueued - t_filtered).count();
            total_filter_time += enqueue_ms;

            std::cout << "Frame " << cloud_index << "/" << total_frames
                      << "  Frame filtering time = " << filter_ms << "ms"
                      << "  Frame to point clouds time = " << enqueue_ms << "ms"
                      << "  queue_depth = " << save_queue.size() << "\n";
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    std::cout << "\n Total capture time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms";
    std::cout << "\n Total filtering time: " << total_filter_time << "ms \n";

    pipe.stop();
    std::cout << "Capture done. Waiting for frame queue to finish ("
              << save_queue.size() << " frames remaining)...\n";

    save_thread_running = false;
    saver.join();

    std::cout << "Done. Saved " << cloud_index << " point clouds to "
              << result_path << "/\n";
    return EXIT_SUCCESS;
}
catch (const rs2::error& e) {
    std::cerr << "RealSense error: " << e.get_failed_function()
              << " " << e.what() << "\n";
    return EXIT_FAILURE;
}
catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
}
