#include <iostream>
#include <string>
#include <cstring>
#include <filesystem>

#include <fcntl.h>
#include <unistd.h>
#include <termios.h>


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <serial_port>\n";
        return 1;
    }

    const std::string port = argv[1];
    int fd = open(port.c_str(), O_WRONLY | O_NOCTTY | O_SYNC);
    if (fd < 0)
    {
        std::cerr << "Failed to open port: " << strerror(errno) << "\n";
        return 1;
    }

    // Configure port to match receiver settings
    struct termios tty{};
    tcgetattr(fd, &tty);
    cfsetospeed(&tty, B9600);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tcsetattr(fd, TCSANOW, &tty);

    // Rising edge 5
    const uint8_t pulse_byte = '5';
    const uint8_t low_byte   = 0x00;

    // Pulse duration: 0.05 ms
    // TODO: Adjust if needed (5ms to 0.005ms)
    const unsigned int pulse_duration_us = 50;

    std::cout << "Sending TTL pulse (" << pulse_duration_us << " µs)\n";

    write(fd, &pulse_byte, 1);

    // wait until byte is actually sent
    tcdrain(fd);                              

    // hold line high for 0.05 ms
    usleep(pulse_duration_us);               

    // falling edge
    write(fd, &low_byte, 1);                 
    tcdrain(fd);

    std::cout << "Pulse sent.\n";

    close(fd);
    return 0;
}