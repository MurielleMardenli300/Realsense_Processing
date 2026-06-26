#include <librealsense2/rs.hpp>
#include <iostream>
#include <filesystem>
#include <boost/program_options.hpp>

int main(int argc, char *argv[]) try {
    namespace po = boost::program_options;

    // Argument parser for file name input
    po::variables_map vm;
    po::options_description desc("Allowed options");

    std::string experiment_name;
    int fps = 0;

    desc.add_options()
    ("help,h", "show help message")
    ("fps,f", po::value<int>(&fps)->required(),
        "input fps")
    ("experiment_name,e", po::value<std::string>(&experiment_name)->required(),
         "input folder name");
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << '\n';
        return 0;
    }
    po::notify(vm);

    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16,  fps);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, fps);

    std::string video_file = "data/" + experiment_name + ".db3";

    if (!(std::filesystem::exists("data")))
    {
        std::filesystem::create_directories("data");
        std::cout << "Created directory: " << "data" << '\n';
    }

    cfg.enable_record_to_file("data/" + experiment_name + ".db3");
    pipe.start(cfg); 

    int frame_count = 0;
    int total_frames = fps * 10;  // Record for 10 seconds
    for (int i = 0; i < total_frames; i++) { 
        pipe.wait_for_frames();
        frame_count++;
        std::cout << "Captured " << frame_count << "/" << total_frames << std::endl;
    }

    pipe.stop();  // file is finalized here
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}