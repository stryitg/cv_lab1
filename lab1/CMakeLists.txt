project(lab1)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

set(Boost_INCLUDE_DIR "/usr/include/boost/")
set(Boost_LIBRARY_DIR "/usr/lib/")

find_package(Boost 1.72 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

set(OpenCV_INCLUDE_DIR "/usr/include/opencv4/opencv2/")
set(OpenCV_LIBRARY_DIR "/usr/lib")

find_package(OpenCV COMPONENTS core highgui REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIR})

find_package(Threads)

add_executable(lab1 ${SOURCES})
target_link_libraries(lab1 ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
