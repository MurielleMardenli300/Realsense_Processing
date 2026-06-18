#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp> 
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <csignal> 
#include <thread>
#include <format>
#include <filesystem>
#include <boost/program_options.hpp>


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


void save_points(const rs2::points& points, const rs2::video_frame& color, int index, const std::string& experiment_name = "test_murielle")
{
    std::ostringstream filename;
    filename << "../results/" << experiment_name << "/pointcloud_" << std::setw(5) << std::setfill('0') << index << ".ply";

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

void save_points_roi(
    const rs2::points&      points,
    const rs2::video_frame& color,
    int                     index,
    const ROI&              roi,
    const std::string&     experiment_name = "test_murielle")
{
    std::ostringstream filename;

    filename << "../results/" << experiment_name << "/pointcloud_"
             << std::setw(5) << std::setfill('0') << index << ".ply";

    auto vertices   = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();

    int w          = color.get_width();
    int h          = color.get_height();
    auto color_data = reinterpret_cast<const uint8_t*>(color.get_data());

    // ── Reshape flat vertices into 2D grid (H x W) ──
    // Same concept as issue #2769:
    // verts = np.reshape(h, w, 3)
    // roi   = verts[ymin:ymax, xmin:xmax]
    struct Vertex3D { float x, y, z; };
    std::vector<std::vector<Vertex3D>> verts_2d(h, std::vector<Vertex3D>(w));
    std::vector<std::vector<std::pair<float,float>>> tex_2d(
        h, std::vector<std::pair<float,float>>(w));

    for (int row = 0; row < h; row++)
        for (int col = 0; col < w; col++)
        {
            int i = row * w + col;
            verts_2d[row][col] = { vertices[i].x,
                                   vertices[i].y,
                                   vertices[i].z };
            tex_2d[row][col]   = { tex_coords[i].u,
                                   tex_coords[i].v };
        }

    // ── Clamp ROI to frame bounds ──
    int x0 = std::max(roi.xmin, 0);
    int x1 = std::min(roi.xmax, w - 1);
    int y0 = std::max(roi.ymin, 0);
    int y1 = std::min(roi.ymax, h - 1);

    // ── Count valid points inside ROI ──
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

    // ── Write PLY ──
    std::ofstream ofs(filename.str());
    ofs << "ply\nformat ascii 1.0\n"
        << "element vertex " << valid << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "end_header\n";

    for (int row = y0; row < y1; row++)
    {
        for (int col = x0; col < x1; col++)
        {
            const auto& v = verts_2d[row][col];
            if (v.z <= 0) continue;

            // Map texture coordinate to color pixel
            const auto& tc = tex_2d[row][col];
            int cx = std::min(std::max(int(tc.first  * w), 0), w - 1);
            int cy = std::min(std::max(int(tc.second * h), 0), h - 1);
            int px = (cy * w + cx) * 3;

            ofs << v.x << " " << v.y << " " << v.z << " "
                << int(color_data[px + 2]) << " "   // R
                << int(color_data[px + 1]) << " "   // G
                << int(color_data[px + 0]) << "\n"; // B
        }
    }

    std::cout << "Saved " << filename.str()
              << " (" << valid << " pts in ROI ["
              << x0 << "," << y0 << "]-["
              << x1 << "," << y1 << "])\n";
}


int main(int argc, char *argv[]) try
{
    std::signal(SIGINT, signal_handler);
    namespace po = boost::program_options;

    // Argument parser for file name input
    po::variables_map vm;
    po::options_description desc("Allowed options");

    std::string experiment_name;

    desc.add_options()
        ("help,h", "show help message")
        ("experiment_name,f", po::value<std::string>(&experiment_name)->required(),
         "input folder name");
        po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << '\n';
        return 0;
    }
    po::notify(vm);
    std::cout << "Experiment name: " << experiment_name << '\n';

    if (!(std::filesystem::exists("../results/" + experiment_name)))
    {
        std::filesystem::create_directories("../results/" + experiment_name);
        std::cout << "Created directory: " << "../results/" + experiment_name << '\n';
    }
    

    std::string video_file = "capture_breath.db3";
    int sequence_time = 10; // seconds
    int fps = 6;

    // Define ROI around torso center
    int torso_width = 520;

    // For torso from ymin=-0.175m to 0.275m
    // int torso_height = 720*0.65;
    ROI roi;
    roi.xmin = 400;
    roi.xmax = roi.xmin + torso_width;
    roi.ymin = 25;
    roi.ymax = 525;


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
    thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, 0.1f);
    thr_filter.set_option(RS2_OPTION_MAX_DISTANCE, 1.0f);

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

    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16,  6);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 6);

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

            save_points_roi(points, color, cloud_index, roi, experiment_name);
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