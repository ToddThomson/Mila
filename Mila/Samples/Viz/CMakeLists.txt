cmake_minimum_required(VERSION 3.31)
project(Direct2DVisualizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable( MilaViewer WIN32
    "Src/MainWindow.cpp"
)

# Link Win32, Direct2D, DirectWrite, Windows Imaging Component (for bitmap formats)
target_link_libraries( MilaViewer
    d2d1
    windowscodecs
    user32
    gdi32
)

# Ensure Unicode support
target_compile_definitions( MilaViewer PRIVATE UNICODE _UNICODE )

# Optimize by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
