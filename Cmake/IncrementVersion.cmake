# Read the current version from the version file
file(READ "${CMAKE_SOURCE_DIR}/version.txt" VERSION_CONTENTS)
string(STRIP "${VERSION_CONTENTS}" VERSION)

# Split the version into major, minor, and patch
string(REPLACE "." ";" VERSION_LIST ${VERSION})
list(GET VERSION_LIST 0 VERSION_MAJOR)
list(GET VERSION_LIST 1 VERSION_MINOR)
list(GET VERSION_LIST 2 VERSION_PATCH)

# Increment the patch version
math(EXPR VERSION_PATCH "${VERSION_PATCH} + 1")

# Combine the new version
set(NEW_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# Write the new version back to the version file
file(WRITE "${CMAKE_SOURCE_DIR}/version.txt" "${NEW_VERSION}")

# Output the new version
message(STATUS "New version: ${NEW_VERSION}")