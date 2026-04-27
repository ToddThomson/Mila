# CMake project for Mila Samples

#add_subdirectory( MNIST ) FIXME:Re-enable after alpha.5 completed
#add_subdirectory( Bard ) FIXME:Re-enable after alpha.5 completed
add_subdirectory( Chat )

if(WIN32)
  #add_subdirectory(Viz)
else()
  message(STATUS "Skipping Samples/Viz: Windows-only sample (not WIN32)")
endif()
