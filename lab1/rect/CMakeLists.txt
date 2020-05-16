file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(rect STATIC ${SOURCES})

find_package(OpenCV COMPONENTS core highgui imgcodecs imgproc REQUIRED)
target_include_directories(rect PUBLIC ${OpenCV_INCLUDE_DIR})
target_link_directories(rect PUBLIC ${OpenCV_LIBRARY_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_include_directories(rect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rect PRIVATE ${OpenCV_LIBRARIES}
                      PUBLIC Eigen3::Eigen)
