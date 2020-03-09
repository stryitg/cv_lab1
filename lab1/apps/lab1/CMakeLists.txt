file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

add_executable(lab1 ${SOURCES})

target_include_directories(lab1 PRIVATE ${CMAKE_SOURCE_DIR}/shifts/include)
target_link_libraries(lab1 PRIVATE shifts)