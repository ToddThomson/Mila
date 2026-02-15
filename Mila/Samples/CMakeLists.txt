# CMake project for Mila Samples

add_subdirectory( MNIST )
add_subdirectory( Bard )
add_subdirectory( Chat )

if(WIN32)
  #add_subdirectory(Viz)
else()
  message(STATUS "Skipping Samples/Viz: Windows-only sample (not WIN32)")
endif()
