#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <csignal>
#include <thread>
#include <filesystem>
#include <optional>
#include <boost/program_options.hpp>

static bool running = true;

std::string load_camera_config(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Could not open JSON file: " + path);
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

struct ROI {
    int xmin = 100;
    int xmax = 540;
    int ymin = 100;
    int ymax = 380;
};

// ── Interactive ROI selection ────────────────────────────────────────────────
// Pops an OpenCV window showing the given color frame; drag a box, ENTER/SPACE
// to confirm, 'c' (or an empty drag) to cancel and fall back to the full frame.
ROI select_roi_interactive(const rs2::video_frame& color_frame)
{
    int w = color_frame.get_width();
    int h = color_frame.get_height();

    // rs2::video_frame data here is RGB8 (see cfg.enable_stream below) --
    // OpenCV wants BGR for correct on-screen colors.
    cv::Mat rgb(h, w, CV_8UC3, const_cast<void*>(color_frame.get_data()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    std::cout << "  Drag a rectangle over the ROI, then press ENTER/SPACE to confirm "
                 "(or 'c' to cancel and use the full frame).\n";
    cv::Rect r = cv::selectROI("Select ROI (ENTER to confirm, c to cancel)", bgr, false, false);
    cv::destroyAllWindows();

    ROI roi;
    if (r.width == 0 || r.height == 0)
    {
        roi.xmin = 0;
        roi.ymin = 0;
        roi.xmax = w;
        roi.ymax = h;
        std::cout << "  No ROI selected -- using full frame (" << w << "x" << h << ").\n";
        return roi;
    }

    roi.xmin = std::max(0, r.x);
    roi.ymin = std::max(0, r.y);
    roi.xmax = std::min(w, r.x + r.width);
    roi.ymax = std::min(h, r.y + r.height);
    std::cout << "  Selected ROI: x=[" << roi.xmin << "," << roi.xmax
              << "), y=[" << roi.ymin << "," << roi.ymax << ")\n";
    return roi;
}

void save_points_roi(
    const rs2::points&      points,
    const rs2::video_frame& color,
    int                     index,
    const ROI&              roi,
    const std::string&      results_path)
{
    std::ostringstream filename;
    filename << results_path << "/pointcloud_"
             << std::setw(5) << std::setfill('0') << index << ".ply";

    auto vertices   = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();

    int w          = color.get_width();
    int h          = color.get_height();
    auto color_data = reinterpret_cast<const uint8_t*>(color.get_data());

    struct Vertex3D { float x, y, z; };
    std::vector<std::vector<Vertex3D>> verts_2d(h, std::vector<Vertex3D>(w));
    std::vector<std::vector<std::pair<float,float>>> tex_2d(
        h, std::vector<std::pair<float,float>>(w));

    for (int row = 0; row < h; row++)
        for (int col = 0; col < w; col++)
        {
            int i = row * w + col;
            verts_2d[row][col] = { vertices[i].x, vertices[i].y, vertices[i].z };
            tex_2d[row][col]   = { tex_coords[i].u, tex_coords[i].v };
        }

    int x0 = roi.xmin,   x1 = roi.xmax;
    int y0 = roi.ymin,   y1 = roi.ymax;

    size_t valid = 0;
    for (int row = y0; row < y1; row++)
        for (int col = x0; col < x1; col++)
            if (verts_2d[row][col].z > 0)
                valid++;
        

    if (valid == 0)
    {
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
        for (int col = x0; col < x1; col++)
        {
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


// Returns the number of point clouds saved, or -1 on error.
// `same_roi`: if true, reuse `shared_roi` across files instead of asking every time.
int process_file(
    const std::filesystem::path& video_path,
    const std::string&           results_path,
    bool                         same_roi,
    std::optional<ROI>&          shared_roi)
{
    std::cout << "\n──────────────────────────────────────────\n";
    std::cout << "Processing: " << video_path.filename() << "\n";
    std::cout << "Output dir: " << results_path << "\n";
    std::filesystem::create_directories(results_path);

    rs2::context ctx;

    // ── Camera filters ────────────────────────────────────────────────────────
    rs2::threshold_filter    thr_filter;
    rs2::disparity_transform depth_to_disparity(true);
    rs2::spatial_filter      spat_filter;
    rs2::temporal_filter     temp_filter;
    rs2::disparity_transform disparity_to_depth(false);
    rs2::hole_filling_filter hole_filter;

    thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.8f);
    thr_filter.set_option(RS2_OPTION_MAX_DISTANCE, 1.7f);
    spat_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE,    2);
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5f);
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
    spat_filter.set_option(RS2_OPTION_HOLES_FILL,          0);
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
    temp_filter.set_option(RS2_OPTION_HOLES_FILL,          0);

    // ── Pipeline ──────────────────────────────────────────────────────────────
    rs2::pipeline pipe(ctx);
    rs2::config   cfg;
    cfg.enable_device_from_file(video_path.string(), false);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16,  15);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 15);

    auto profile  = pipe.start(cfg);
    auto playback = profile.get_device().as<rs2::playback>();
    playback.set_real_time(false);

    // ── End-of-file detection via status callback ─────────────────────────────
    bool file_done{false};
    playback.set_status_changed_callback([&file_done](rs2_playback_status status)
    {
        if (status == RS2_PLAYBACK_STATUS_STOPPED)
        {
            file_done = true;
            std::cout << "  Playback finished (EOF).\n";
        }
    });

    rs2::pointcloud pc;
    rs2::points     points;
    rs2::align      align_to_color(RS2_STREAM_COLOR);

    // ── Auto-exposure warm-up ─────────────────────────────────────────────────
    const int max_warmup  = 60;
    const int stable_need = 5;
    int   stable_count    = 0;
    float last_exposure   = -1.0f;

    for (int wi = 0; wi < max_warmup && running && !file_done; wi++)
    {
        rs2::frameset frames;
        if (!pipe.poll_for_frames(&frames)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto color = frames.get_color_frame();
        if (!color) continue;

        if (color.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE))
        {
            float exp = static_cast<float>(
                color.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE));
            if (last_exposure > 0 && std::abs(exp - last_exposure) / last_exposure < 0.02f)
                stable_count++;
            else
                stable_count = 0;
            last_exposure = exp;
            if (stable_count >= stable_need) {
                std::cout << "  Auto-exposure stable after " << wi << " warm-up frames.\n";
                break;
            }
        }
        else
        {
            if (wi >= 10) break;  // metadata unavailable — fixed skip
        }
    }

    // ── Grab one aligned frame to select the ROI on ───────────────────────────
    rs2::frameset first_aligned;
    bool got_first = false;
    for (int i = 0; i < 200 && running && !file_done && !got_first; i++)
    {
        rs2::frameset frames;
        if (!pipe.poll_for_frames(&frames)) {
            if (playback.current_status() == RS2_PLAYBACK_STATUS_STOPPED) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto aligned = align_to_color.process(frames);
        if (aligned.get_color_frame() && aligned.get_depth_frame())
        {
            first_aligned = aligned;
            got_first = true;
        }
    }

    if (!got_first)
    {
        std::cout << "  Could not get a frame to select ROI on, skipping file.\n";
        pipe.stop();
        return -1;
    }

    ROI roi;
    if (same_roi && shared_roi.has_value())
    {
        roi = *shared_roi;
        std::cout << "  Reusing shared ROI: x=[" << roi.xmin << "," << roi.xmax
                  << "), y=[" << roi.ymin << "," << roi.ymax << ")\n";
    }
    else
    {
        roi = select_roi_interactive(first_aligned.get_color_frame());
        if (same_roi) shared_roi = roi;
    }

    // helper to filter depth + build/save one point cloud for a given aligned frameset
    int cloud_index = 0;
    auto process_one = [&](const rs2::frameset& aligned)
    {
        auto color = aligned.get_color_frame();
        auto depth = aligned.get_depth_frame();
        if (!color || !depth) return;

        rs2::frame filtered = depth;
        filtered = thr_filter.process(filtered);
        filtered = spat_filter.process(filtered);
        filtered = temp_filter.process(filtered);

        cloud_index++;
        pc.map_to(color);
        points = pc.calculate(filtered);
        save_points_roi(points, color, cloud_index, roi, results_path);
    };

    // process the frame we used for ROI selection too, don't waste it
    process_one(first_aligned);

    // ── Extraction ──────────────────────────────────────────────────────────
    while (running && !file_done)
    {
        rs2::frameset frames;
        if (!pipe.poll_for_frames(&frames)) {
            if (playback.current_status() == RS2_PLAYBACK_STATUS_STOPPED)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto aligned = align_to_color.process(frames);
        process_one(aligned);
    }

    pipe.stop();
    std::cout << "  Done: " << cloud_index << " point clouds saved.\n";
    return cloud_index;
}


int main(int argc, char* argv[]) try
{
    namespace po = boost::program_options;

    po::variables_map vm;
    po::options_description desc("Allowed options");

    std::string dir;
    bool same_roi = false;
    desc.add_options()
        ("help,h", "show help message")
        ("dir,d", po::value<std::string>(&dir)->required(),
         "sub-directory inside data/ containing the .db3 files to process")
        ("same-roi", po::bool_switch(&same_roi),
         "select the ROI once (on the first file) and reuse it for every subsequent file");

    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << '\n'; return 0; }
    po::notify(vm);

    // ── Locate all .db3 files under data/<dir>/ ───────────────────────────────
    std::filesystem::path data_dir = std::filesystem::path("data") / dir;
    if (!std::filesystem::exists(data_dir))
        throw std::runtime_error("Directory not found: " + data_dir.string());

    std::vector<std::filesystem::path> db3_files;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir))
        if (entry.is_regular_file() && entry.path().extension() == ".db3")
            db3_files.push_back(entry.path());

    if (db3_files.empty())
        throw std::runtime_error("No .db3 files found in " + data_dir.string());

    std::sort(db3_files.begin(), db3_files.end());  // alphabetical / deterministic order
    std::cout << "Found " << db3_files.size() << " .db3 file(s) in " << data_dir << "\n";
    if (same_roi)
        std::cout << "ROI will be selected once and reused for all files.\n";
    else
        std::cout << "You will be asked to select an ROI for each file.\n";

    std::optional<ROI> shared_roi;
    int total_clouds = 0;
    int files_done   = 0;

    for (const auto& video_path : db3_files)
    {
        if (!running) break;

        std::string experiment_name = video_path.stem().string();
        std::string results_path =
            (std::filesystem::path("results") / dir / experiment_name).string();

        int n = process_file(video_path, results_path, same_roi, shared_roi);
        if (n >= 0) { total_clouds += n; files_done++; }
    }

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << "Processed " << files_done << " / " << db3_files.size() << " file(s), "
              << total_clouds << " point clouds saved in total.\n";

    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error: " << e.get_failed_function()
              << "(" << e.get_failed_args() << ")\n" << e.what() << "\n";
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
}