# The mila-app target's steps, run as a script: stage the session the window carries, then package the window.
# Given MILA_BUILD_DIRECTORY, MILA_WINDOW_DIRECTORY, MILA_NPM and MILA_PACKAGE_SCRIPT by the target.

set( session_directory "${MILA_WINDOW_DIRECTORY}/session" )

# A fresh copy each time, so a library the session no longer loads is not carried on from an earlier staging.
file( REMOVE_RECURSE "${session_directory}" )

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${MILA_BUILD_DIRECTORY}" --component mila-session --prefix "${session_directory}"
    COMMAND_ERROR_IS_FATAL ANY
)

# The window's packages are installed once; their versions are the lock file's.
if( NOT EXISTS "${MILA_WINDOW_DIRECTORY}/node_modules" )
    execute_process(
        COMMAND "${MILA_NPM}" ci
        WORKING_DIRECTORY "${MILA_WINDOW_DIRECTORY}"
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()

execute_process(
    COMMAND "${MILA_NPM}" run "${MILA_PACKAGE_SCRIPT}"
    WORKING_DIRECTORY "${MILA_WINDOW_DIRECTORY}"
    COMMAND_ERROR_IS_FATAL ANY
)
