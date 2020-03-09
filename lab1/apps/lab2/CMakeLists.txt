file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

add_executable(lab2 ${SOURCES})

target_include_directories(lab2 PRIVATE 
                          ${CMAKE_SOURCE_DIR}/shifts/include
                          ${CMAKE_SOURCE_DIR}/fundamental_matrix/include)
target_link_libraries(lab2 PRIVATE shifts fundamental_matrix)