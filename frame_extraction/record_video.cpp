#include <librealsense2/rs.hpp>
#include <iostream>

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);

    cfg.enable_record_to_file("../test.db3");
    pipe.start(cfg);  // recording starts here

    int frame_count = 0;
    for (int i = 0; i < 300; i++) {  // ~10 seconds at 30fps
        pipe.wait_for_frames();
        frame_count++;
    }
    std::cout << "Captured " << frame_count << " frames." << std::endl;

    pipe.stop();  // file is finalized here
    return 0;
} 