# Mila, the assistant app (Specifications/MilaApp.md): the session process is a C++ target; the window and page are
# Node, packaged by Electron's builder. mila-app joins the two so the app builds as one target.
add_subdirectory( Session )

# Not part of ALL: packaging takes about a minute and writes an installer of about 525 MB, wanted only when asked for.
find_program( MILA_NPM NAMES npm.cmd npm )

if( WIN32 )
    set( package_script "build:windows" )
else()
    set( package_script "build:linux" )
endif()

if( MILA_NPM )
    add_custom_target( mila-app
        COMMAND ${CMAKE_COMMAND}
            -D "MILA_BUILD_DIRECTORY=${CMAKE_BINARY_DIR}"
            -D "MILA_WINDOW_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}/Window"
            -D "MILA_NPM=${MILA_NPM}"
            -D "MILA_PACKAGE_SCRIPT=${package_script}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/PackageApp.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Window"
        USES_TERMINAL
        COMMENT "Packaging the Mila app into Mila/Apps/Assistant/Window/dist"
        VERBATIM
    )
else()
    add_custom_target( mila-app
        COMMAND ${CMAKE_COMMAND} -E echo "mila-app needs Node.js: install it, put npm on the PATH and configure again."
        COMMAND ${CMAKE_COMMAND} -E false
        VERBATIM
    )
endif()

add_dependencies( mila-app mila-session )
