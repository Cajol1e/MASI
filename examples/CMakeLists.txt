cmake_minimum_required(VERSION 3.1) # for CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD 11)

include(GNUInstallDirs)
macro(flexnn_install name)
    install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endmacro()

if(MSVC)
    # warning C4018: '<': signed/unsigned mismatch
    # warning C4244: 'argument': conversion from 'uint64_t' to 'const unsigned int', possible loss of data
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4018 /wd4244 /wd4996)
endif()

add_executable(layertest layertest.cpp)
target_link_libraries(layertest PRIVATE ncnn)

add_executable(flexnndemo flexnndemo.cpp)
target_link_libraries(flexnndemo PRIVATE ncnn)


# add all examples to a virtual project group
set_property(TARGET flexnndemo PROPERTY FOLDER "examples")
set_property(TARGET layertest PROPERTY FOLDER "examples")


flexnn_install(flexnndemo)
flexnn_install(layertest)