file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../Version.txt" VERSION_CONTENT)

# Parse version components
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)-([a-zA-Z]+)\\.([0-9]+)" 
       VERSION_MATCH "${VERSION_CONTENT}")

if(VERSION_MATCH)
    set(MAJOR ${CMAKE_MATCH_1})
    set(MINOR ${CMAKE_MATCH_2})
    set(BUILD ${CMAKE_MATCH_3})
    set(TAG ${CMAKE_MATCH_4})
    set(TAG_VERSION ${CMAKE_MATCH_5})
    
    math(EXPR NEW_BUILD "${BUILD} + 1")
    
    set(NEW_VERSION "${MAJOR}.${MINOR}.${NEW_BUILD}-${TAG}.${TAG_VERSION}")
    
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/../Version.txt" "${NEW_VERSION}")
    
    message(STATUS "Version updated to ${NEW_VERSION}")
else()
    message(FATAL_ERROR "Failed to parse version from Version.txt")
endif()