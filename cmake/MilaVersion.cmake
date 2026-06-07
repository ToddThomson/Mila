# Mila version helpers.

# mila_read_version(<version_file> <out_numeric> <out_prerelease>)
#
# Reads a SemVer string (X.Y.Z or X.Y.Z-PRERELEASE, e.g. "0.13.39-alpha.5") from
# version_file and returns, via the named output variables:
#   out_numeric    -- the numeric major.minor.patch triple, suitable for
#                     project(VERSION ...).
#   out_prerelease -- the optional prerelease label (e.g. "alpha.5"), empty if
#                     absent. It cannot live in project(VERSION) -- CMake accepts
#                     only numeric components -- so callers carry it separately.
#
# Aborts with FATAL_ERROR if the file does not match the expected format.
function(mila_read_version version_file out_numeric out_prerelease)
    file(READ "${version_file}" _raw)
    string(STRIP "${_raw}" _raw)

    if(NOT _raw MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(-(.+))?$")
        message(FATAL_ERROR "Version file '${version_file}': expected X.Y.Z or X.Y.Z-PRERELEASE, got '${_raw}'")
    endif()

    set(${out_numeric} "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" PARENT_SCOPE)
    set(${out_prerelease} "${CMAKE_MATCH_5}" PARENT_SCOPE)
endfunction()
