file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(shifts STATIC ${SOURCES})

target_include_directories(shifts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Boost 1.72 COMPONENTS program_options REQUIRED)
target_include_directories(shifts PRIVATE ${Boost_INCLUDE_DIR})
target_link_directories(shifts PRIVATE ${Boost_LIBRARY_DIR})

find_package(OpenCV COMPONENTS core highgui imgcodecs imgproc REQUIRED)
target_include_directories(shifts PUBLIC ${OpenCV_INCLUDE_DIR})
target_link_directories(shifts PUBLIC ${OpenCV_LIBRARY_DIR})

find_package(Threads)

target_link_libraries(shifts PRIVATE ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} 
                             PUBLIC ${OpenCV_LIBRARIES})

