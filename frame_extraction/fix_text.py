int main() try
{
    std::signal(SIGINT, signal_handler);

    int sequence_time = 10; // seconds
    int fps = 6;

    // Define ROI around torso center
    int torso_width = 1280 * 0.45;
    int torso_height = 720 * 0.65;
    ROI roi;
    roi.xmin = (1280 - torso_width) / 2;
    roi.xmax = roi.xmin + torso_width;
    roi.ymin = (720 - torso_height) / 2;
    roi.ymax = roi.ymin + torso_height;

    // 1. Core Native Context Setup
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0)
        throw std::runtime_error("No RealSense device detected.");

    rs2::device dev = devices[0];

    // 2. Safely handle Advanced Mode Setup 
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
    std::string json_content = load_camera_config("../../camera_settings/manual_exp_settings.json");
    advanced_mode.load_json(json_content);
    std::cout << "Settings loaded.\n";

    // 3. Initialize Pipeline and Configure Streams
    rs2::pipeline pipe(ctx); // Pass the context explicitly
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, fps);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, fps);

    // Declare pointcloud utility outside the loop
    rs2::pointcloud pc;
    rs2::points points;

    // 4. Start the pipeline within a scope-safe block
    std::cout << "Starting stream pipeline...\n";
    rs2::pipeline_profile profile = pipe.start(cfg);

    int index = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (running)
    {
        // Add a timeout handle so frames don't deadlock if hardware drops
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 5000)) // 5-second timeout safeguard
        {
            std::cout << "Warning: Frame timeout reached. Hardware dropped connection.\n";
            break;
        }

        auto depth_frame = frames.get_depth_frame();
        auto color_frame = frames.get_color_frame();

        if (!depth_frame || !color_frame)
            continue;

        // Apply filters here...
        // e.g., depth_frame = thr_filter.process(depth_frame);

        // Map texture and generate pointcloud data
        pc.map_to(color_frame);
        points = pc.calculate(depth_frame);

        // Save data out
        save_points_roi(points, color_frame, index++, roi);

        // Break loop when sequence time limits are met
        auto current_time = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() >= sequence_time)
        {
            std::cout << "Sequence time limit reached.\n";
            break;
        }
    }

    // 5. THE CRITICAL FIX: Clean teardown sequence
    std::cout << "Shutting down pipeline cleanly...\n";
    pipe.stop(); 
    std::cout << "Pipeline stopped successfully. Device handles freed.\n";

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
