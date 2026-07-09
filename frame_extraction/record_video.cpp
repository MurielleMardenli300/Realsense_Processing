#include <librealsense2/rs.hpp>
#include <iostream>
#include <fstream>

#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp> 
#include <filesystem>
#include <boost/program_options.hpp>
#include <csignal> 
#include <thread>
#include <format>

// TTL 
#include <termios.h>   // POSIX 
#include <fcntl.h>     // open(), O_RDONLY
#include <unistd.h>    // read(), close()

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


int open_trigger_port(const std::string& port, int baud = B9600)
{
    int fd = open(port.c_str(), O_RDONLY | O_NOCTTY | O_SYNC);
    if (fd < 0)
        throw std::runtime_error("Could not open trigger port: " + port +
                                 " (" + strerror(errno) + ")");

    struct termios tty{};
    if (tcgetattr(fd, &tty) != 0)
        throw std::runtime_error("tcgetattr failed on " + port);

    cfsetispeed(&tty, baud);   // input baud rate
    cfsetospeed(&tty, baud);   // output baud rate (not used but required)

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8-bit chars
    tty.c_cflag |= (CLOCAL | CREAD); // ignore modem controls
    tty.c_cflag &= ~(PARENB | PARODD); // no parity
    tty.c_cflag &= ~CSTOPB; // 1 stop bit
    tty.c_cflag &= ~CRTSCTS; // no hardware flow control

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); 
    tty.c_iflag &= ~(IGNBRK | BRKINT | ICRNL | INLCR | PARMRK | INPCK | ISTRIP);

    tty.c_lflag = 0;   // raw mode — no echo, no signals
    tty.c_oflag = 0;   // raw output

    // Blocking read: wait indefinitely for at least 1 byte
    tty.c_cc[VMIN]  = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0)
        throw std::runtime_error("tcsetattr failed on " + port);

    return fd;
}


// Most MRI trigger interfaces send a single ASCII byte
// when the TTL goes high. We accept any byte as a valid trigger.
// Returns the byte received, or -1 if interrupted by signal.
int wait_for_ttl_trigger(int fd)
{
    std::cout << "Waiting for TTL trigger...\n";
    uint8_t byte = 0;
    while (running)
    {
        // Use select() with a short timeout so we can check `running` periodically
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(fd, &read_fds);

        struct timeval timeout;
        timeout.tv_sec  = 0;

        // 0.01 ms polling interval 
        // TODO: Adjust according to pulse length (0.05ms for Philips Ingenia?)
        timeout.tv_usec = 10;  

        int ret = select(fd + 1, &read_fds, nullptr, nullptr, &timeout);
        if (ret < 0)
        {
            // interrupted by keyboard 
            if (errno == EINTR) continue; 
            throw std::runtime_error("select() failed on trigger port");
        }

        // timeout no byte yet, loop and re-check
        if (ret == 0) continue; 

        // A byte is available
        ssize_t n = read(fd, &byte, 1);
        if (n > 0 && byte != 0x00)   // ignore the falling-edge null byte (end of pulse)
        {
            std::cout << "TTL trigger received! (byte=0x"
                      << std::hex << (int)byte << std::dec << ")\n";
            std::cout << "TTL STOP";
            return byte;
        }
    }
    return -1; 
}



int main(int argc, char *argv[]) try {
    namespace po = boost::program_options;

    po::variables_map vm;
    po::options_description desc("Allowed options");

    std::string name;
    std::string dir = "";
    std::string trigger_port;

    int fps = 0;
    int length = 20;

    desc.add_options()
    ("help,h", "show help message")
    ("fps,f", po::value<int>(&fps)->required(),
        "input fps")
    ("name,e", po::value<std::string>(&name)->required(),
         "input folder name")
    ("length,f", po::value<int>(&length),
        "input length") 
    ("dir,f", po::value<std::string>(&dir),
        "input dir") 
    ("trigger-port,t", po::value<std::string>(&trigger_port)->default_value(""),
                          "serial port for TTL trigger (ex /dev/ttyUSB0). "
                          "If empty, recording starts immediately.")
    ;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << '\n';
        return 0;
    }
    po::notify(vm);


    // Camera setup

    //     rs2::context ctx;
    // rs2::device_list devices = ctx.query_devices();
    // if (devices.size() == 0)
    //     throw std::runtime_error("No RealSense device detected.");

    // rs2::device dev = devices[0];

    // // Enable advanced mode for json
    // if (!dev.is<rs400::advanced_mode>())
    //     throw std::runtime_error("Device does not support advanced mode.");

    // auto advanced_mode = dev.as<rs400::advanced_mode>();

    // if (!advanced_mode.is_enabled())
    // {
    //     std::cout << "Enabling advanced mode...\n";
    //     advanced_mode.toggle_advanced_mode(true);

    //     std::this_thread::sleep_for(std::chrono::seconds(3));
    //     devices = ctx.query_devices();
    //     dev = devices[0];
    //     advanced_mode = dev.as<rs400::advanced_mode>();
    // }

    // std::cout << "Loading camera settings from JSON...\n";
    // std::string json_content = load_camera_config("../camera_settings/manual_exp_roi.json");
    // advanced_mode.load_json(json_content);
    // std::cout << "Settings loaded.\n";

    // -- Video processing setup

    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16,  fps);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, fps);


    std::string path = "data/" + dir + "/";
    std::string video_file = path + name + ".db3";

    if (!(std::filesystem::exists(path)))
    {
        std::filesystem::create_directories(path);
        std::cout << "Created directory: " << path << '\n';
    }

    cfg.enable_record_to_file(video_file);
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    pipe.start(cfg); 
    
    // -- TTL trigger
    
    
    int trigger_file = -1;
    if (!trigger_port.empty())
    {
        trigger_file = open_trigger_port(trigger_port);
        int result = wait_for_ttl_trigger(trigger_file);
        // Calculate time elapsed between trigger reception and first frame capture
        start = std::chrono::high_resolution_clock::now();
        
        std::cout << "result is: " << result << "\n";
        if (result < 0)
        {
            std::cout << "Interrupted before ttl trigger, exiting.\n";
            close(trigger_file);
            return EXIT_SUCCESS;
        }

        close(trigger_file);

    }
    else
    {
        std::cout << "No trigger port specified, starting immediately.\n";
    }
    
    
    // -- Video recording
    
    std::cout << "Starting reccording\n";
    int frame_count = 0;
    int total_frames = fps * length;  // Record for X seconds
    for (int i = 0; i < total_frames; i++) { 
        pipe.wait_for_frames();

        if(i == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "-- Time elapsed between trigger and 1st recording (ms): " << elapsed.count() << "\n";
        }

        frame_count++;
        std::cout << "Captured " << frame_count << "/" << total_frames << std::endl;
    }

    pipe.stop(); 
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}