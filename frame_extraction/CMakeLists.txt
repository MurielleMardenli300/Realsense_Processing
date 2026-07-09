cmake_minimum_required(VERSION 3.1)
project(my_realsense)

find_package(realsense2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

option(RECORD OFF)
option(EXTRACT OFF)

if(RECORD)
    add_executable(my_recorder record_video.cpp)
    target_link_libraries(my_recorder
    realsense2
    Boost::program_options
)
endif()

if(EXTRACT)
    add_executable(my_extracter video_to_frame.cpp)
    target_link_libraries(my_extracter
    realsense2
    Boost::program_options
)

add_executable(trigger_ttl trigger_ttl.cpp)
    target_link_libraries(trigger_ttl
    realsense2
)

add_executable(live_extract live_frame_extraction.cpp)
    target_link_libraries(live_extract
    realsense2
    Boost::program_options
)

endif()
