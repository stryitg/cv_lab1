file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

add_executable(lab3 ${SOURCES})

target_include_directories(lab3 PRIVATE 
                          ${CMAKE_SOURCE_DIR}/shifts/include
                          ${CMAKE_SOURCE_DIR}/fundamental_matrix/include
                          ${CMAKE_SOURCE_DIR}/rect/include)
target_link_libraries(lab3 PRIVATE shifts fundamental_matrix rect)