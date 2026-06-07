# CMake project for Mila Samples

#add_subdirectory( MNIST ) FIXME:Re-enable after alpha.5 completed
#add_subdirectory( Bard ) FIXME:Re-enable after alpha.5 completed
add_subdirectory( Chat )

# QuickStart is intentionally NOT add_subdirectory()'d here. It is a standalone
# find_package(Mila) example/consumer that must be configured on its own against an
# install prefix, not built in-tree (an in-tree add_subdirectory would call
# find_package(Mila) before Mila is installed, and would link the in-tree target
# rather than the installed package). The Tests/Packaging gate drives it. See
# QuickStart/README.md.

if(WIN32)
  #add_subdirectory(Viz)
else()
  message(STATUS "Skipping Samples/Viz: Windows-only sample (not WIN32)")
endif()
