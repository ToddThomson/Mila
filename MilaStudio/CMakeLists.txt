cmake_minimum_required(VERSION 3.28)

project(MilaViewer
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Windows 11 features
add_compile_definitions(
    UNICODE
    _UNICODE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    WINVER=0x0A00      # Windows 10/11
    _WIN32_WINNT=0x0A00
)

# Find dependencies
find_package(Mila REQUIRED)

# Source files
set(VIEWER_SOURCES
    Src/Main.cpp
    Src/Application.ixx
    Src/MainWindow.ixx
    Src/ModelTreeView.ixx
    Src/DocumentView.ixx
    Src/OutputView.ixx
    Src/Resources.rc
)

# Create executable
add_executable(MilaViewer WIN32 ${VIEWER_SOURCES})

# Link libraries
target_link_libraries(MilaViewer
    PRIVATE
        Mila::Mila
        comctl32
        uxtheme
        dwmapi
        d2d1
        dwrite
)

# Set output directory
set_target_properties(MilaViewer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Copy resources
add_custom_command(TARGET MilaViewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/Resources"
        "$<TARGET_FILE_DIR:MilaViewer>/Resources"
)