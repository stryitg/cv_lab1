file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

add_executable(lab5 "${SOURCES}")

find_package(Boost 1.72 COMPONENTS program_options REQUIRED)
target_include_directories(lab5 PRIVATE ${Boost_INCLUDE_DIR})
target_link_directories(lab5 PRIVATE ${Boost_LIBRARY_DIR})

find_package(OpenCV COMPONENTS core highgui imgcodecs imgproc REQUIRED)
target_include_directories(lab5 PRIVATE ${OpenCV_INCLUDE_DIR})
target_link_directories(lab5 PRIVATE ${OpenCV_LIBRARY_DIR})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(lab5 PRIVATE ${Boost_LIBRARIES} 
                           PRIVATE ${OpenCV_LIBRARIES}
                           PRIVATE Eigen3::Eigen)

