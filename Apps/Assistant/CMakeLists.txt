# Mila Assistant (Specifications/Apps/MilaAssistant.md): the session process is a C++ target; the window and page are
# Node, packaged by Electron's builder. mila-app joins the two so the app builds as one target, and mila-app-store
# builds the package a Store is given the same way, so neither can carry a session staged by an earlier build.
add_subdirectory( Session )

# Not part of ALL: packaging takes about a minute and writes a package of about 525 to 650 MB, wanted only when asked for.
find_program( MILA_NPM NAMES npm.cmd npm )

if( WIN32 )
    set( package_script "build:windows" )
    set( store_package_script "build:windows:store" )
elseif( APPLE )
    set( package_script "build:mac" )
    set( store_package_script "" )
else()
    set( package_script "build:linux" )
    set( store_package_script "" )
endif()

function( mila_add_package_target target script comment )
    if( NOT MILA_NPM )
        add_custom_target( ${target}
            COMMAND ${CMAKE_COMMAND} -E echo "${target} needs Node.js: install it, put npm on the PATH and configure again."
            COMMAND ${CMAKE_COMMAND} -E false
            VERBATIM
        )
    elseif( script STREQUAL "" )
        add_custom_target( ${target}
            COMMAND ${CMAKE_COMMAND} -E echo "${target}: this platform's Store package is not built yet (BACKLOG.md, Mila Assistant and the Stores)."
            COMMAND ${CMAKE_COMMAND} -E false
            VERBATIM
        )
    else()
        add_custom_target( ${target}
            COMMAND ${CMAKE_COMMAND}
                -D "MILA_BUILD_DIRECTORY=${CMAKE_BINARY_DIR}"
                -D "MILA_WINDOW_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}/Window"
                -D "MILA_NPM=${MILA_NPM}"
                -D "MILA_PACKAGE_SCRIPT=${script}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/PackageApp.cmake"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Window"
            USES_TERMINAL
            COMMENT "${comment}"
            VERBATIM
        )
    endif()

    add_dependencies( ${target} mila-session )
endfunction()

mila_add_package_target( mila-app "${package_script}" "Packaging Mila Assistant into Apps/Assistant/Window/dist" )
mila_add_package_target( mila-app-store "${store_package_script}" "Packaging Mila Assistant for the Store into Apps/Assistant/Window/dist" )
