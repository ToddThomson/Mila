# CMakeList.txt : CMake project for FrontendCuda, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.20)

add_executable( 
    sample_rnn 
    rnn_example.cpp rnn_example.h
    fp16_emu.cpp fp16_emu.h
    init_data.cu init_data.cuh  )

set_property( TARGET sample_rnn PROPERTY CXX_STANDARD 17 )
set_target_properties( sample_rnn PROPERTIES CUDA_ARCHITECTURES "50;60;75;80" )

include_directories( 
    sample_rnn ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

target_link_libraries( 
    sample_rnn
     ${CUDA_LIBRARIES} ${CUDNN_LIBRARY} )