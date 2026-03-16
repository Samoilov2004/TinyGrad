cmake_minimum_required(VERSION 3.16)

project(tinygradc VERSION 0.3.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug
    Release
    RelWithDebInfo
    MinSizeRel
)

function(tg_enable_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(tinygradc SHARED
    src/tg.c
    src/tensor.c
    src/arena.c
    src/autograd.c
    src/ops_basic.c
    src/ops_nn.c
    src/ops_loss.c
    src/optim.c
    src/io.c
    src/nn_layers.c
    src/data.c
    src/train_utils.c
)

target_include_directories(tinygradc PUBLIC include)

if(NOT MSVC)
    target_link_libraries(tinygradc PUBLIC m)
endif()

target_include_directories(tinygradc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

tg_enable_warnings(tinygradc)

add_executable(xor
    examples/xor.c
)
target_link_libraries(xor PRIVATE tinygradc)
tg_enable_warnings(xor)

add_executable(logreg_csv
    examples/logreg_csv.c
)
target_link_libraries(logreg_csv PRIVATE tinygradc)
tg_enable_warnings(logreg_csv)

add_executable(gradcheck
    tests/gradcheck.c
)
target_link_libraries(gradcheck PRIVATE tinygradc)
tg_enable_warnings(gradcheck)

add_executable(xor_mlp
    examples/xor_mlp.c
)
target_link_libraries(xor_mlp PRIVATE tinygradc)
tg_enable_warnings(xor_mlp)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME gradcheck COMMAND gradcheck)
endif()
