#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp> 
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <csignal> 
#include <thread>

static bool running = true;

void signal_handler(int signum)
{
    std::cout << "\nInterrupt received, stopping.\n";
    running = false;
}

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
    int xmin = 150;
    int xmax = 490;
    int ymin = 100;
    int ymax = 380;
};


void save_points(const rs2::points& points, const rs2::video_frame& color, int index)
{
    std::ostringstream filename;
    filename << "../results/test_murielle/pointcloud_" << std::setw(5) << std::setfill('0') << index << ".ply";

    std::cout << "Saving " << filename.str() << " with " << points.size() << " points...\n";

    auto vertices  = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();

    int w = color.get_width();
    int h = color.get_height();
    auto color_data = reinterpret_cast<const uint8_t*>(color.get_data());

    size_t valid = 0;
    for (size_t i = 0; i < points.size(); i++)
        if (vertices[i].z > 0) valid++;

    std::ofstream ofs(filename.str());
    ofs << "ply\nformat ascii 1.0\n"
        << "element vertex " << valid << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "end_header\n";

    for (size_t i = 0; i < points.size(); i++)
    {
        if (vertices[i].z <= 0) continue;

        // Map texture coordinate to color pixel
        int cx = std::min(std::max(int(tex_coords[i].u * w), 0), w - 1);
        int cy = std::min(std::max(int(tex_coords[i].v * h), 0), h - 1);
        int pixel = (cy * w + cx) * 3;

        ofs << vertices[i].x << " "
            << vertices[i].y << " "
            << vertices[i].z << " "
            << int(color_data[pixel + 2]) << " "  // R
            << int(color_data[pixel + 1]) << " "  // G
            << int(color_data[pixel + 0]) << "\n"; // B
    }

    std::cout << "Saved " << filename.str() << " (" << valid << " points)\n";
}


int main() try
{
    std::signal(SIGINT, signal_handler);


    std::string video_file = "capture_breath.db3";
    int sequence_time = 10; // seconds
    int fps = 6;

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

    std::cout << "Loading camera settings from JSON...\n";
    std::string json_content = load_camera_config("../../camera_settings/threshold_settings.json");
    advanced_mode.load_json(json_content);
    std::cout << "Settings loaded.\n";

    // Declare all filters
    // rs2::decimation_filter  dec_filter;
    rs2::threshold_filter   thr_filter;
    rs2::disparity_transform depth_to_disparity(true);   // depth -> disparity
    rs2::spatial_filter     spat_filter;
    rs2::temporal_filter    temp_filter;
    rs2::disparity_transform disparity_to_depth(false);  // disparity -> depth
    rs2::hole_filling_filter hole_filter;

    // Tune filter options

    // dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);

    // Threshold: depth clip in metres
    thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.0f);
    thr_filter.set_option(RS2_OPTION_MAX_DISTANCE, 0.8f);

    // Spatial: edge-preserving smoothing
    spat_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE,   2);  
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5f);
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20); 
    spat_filter.set_option(RS2_OPTION_HOLES_FILL,          0); 

    // Temporal: reduces temporal noise across frames
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f); 
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20); 
    // Persistency mode (0=disabled, 1–8 increasing aggressiveness)
    temp_filter.set_option(RS2_OPTION_HOLES_FILL,          0);


    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16,  6);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 6);

    cfg.enable_record_to_file(video_file);

    pipe.start(cfg);  

    rs2::pointcloud pc;
    rs2::points     points;
    rs2::align      align_to_color(RS2_STREAM_COLOR);

    using clock = std::chrono::steady_clock;
    auto last_saved = clock::now();
    const auto interval = std::chrono::milliseconds(100); // 10 fps = 100 ms per frame

    int cloud_index = 0;
    int total_frames = sequence_time * fps;

    std::cout << "Recording to " << video_file << " and extracting point clouds at 6fps...\n";

        while (running)
    {
        rs2::frameset frames;
        if (!pipe.poll_for_frames(&frames)) continue;

        auto aligned = align_to_color.process(frames);
        auto color   = aligned.get_color_frame();
        auto depth   = aligned.get_depth_frame();

        if (!color || !depth) continue;

        // Apply post processingfilter chain
        rs2::frame filtered = depth;
        // filtered = dec_filter.process(filtered);
        filtered = thr_filter.process(filtered);
        filtered = depth_to_disparity.process(filtered); // convert back
        filtered = spat_filter.process(filtered);
        filtered = temp_filter.process(filtered);
        filtered = disparity_to_depth.process(filtered);  // convert back
        filtered = hole_filter.process(filtered);

        auto now = clock::now();
        if (now - last_saved >= interval && cloud_index < total_frames)
        {
            cloud_index++;
            pc.map_to(color);
            points = pc.calculate(filtered);
            save_points(points, color, cloud_index);
            last_saved = now;
        }
    }

    pipe.stop();
    std::cout << "Done. Saved " << cloud_index << " point clouds.\n";
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