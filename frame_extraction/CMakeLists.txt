cmake_minimum_required(VERSION 3.1)
project(my_recorder)

find_package(realsense2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)


add_executable(my_recorder video_to_frame.cpp)

target_link_libraries(my_recorder
    realsense2
    Boost::program_options
)