file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(fundamental_matrix STATIC ${SOURCES})

find_package(OpenCV COMPONENTS core highgui imgcodecs imgproc REQUIRED)
target_include_directories(fundamental_matrix PUBLIC ${OpenCV_INCLUDE_DIR})
target_link_directories(fundamental_matrix PUBLIC ${OpenCV_LIBRARY_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_include_directories(fundamental_matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fundamental_matrix PRIVATE ${OpenCV_LIBRARIES}
                      PUBLIC Eigen3::Eigen)
