# CMake project for Mila Samples

add_subdirectory(MNIST)
add_subdirectory(CharLM)
#add_subdirectory( Gpt2 )

if(WIN32)
  add_subdirectory(Viz)
else()
  message(STATUS "Skipping Samples/Viz: Windows-only sample (not WIN32)")
endif()
