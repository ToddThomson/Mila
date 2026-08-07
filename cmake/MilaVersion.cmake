# Mila version helpers.

# mila_read_version(<version_file> <out_numeric> <out_prerelease>)
#
# Reads a SemVer string (X.Y.Z or X.Y.Z-PRERELEASE, e.g. "0.20.0-alpha.6.54") from
# version_file and returns, via the named output variables:
#   out_numeric    -- the numeric major.minor.patch triple, suitable for
#                     project(VERSION ...).
#   out_prerelease -- the optional prerelease label (e.g. "alpha.6.54"), empty if
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

# mila_pep440_version(<numeric> <prerelease> <out_version>)
#
# Translates Mila's SemVer into the PEP 440 spelling PyPI requires, so the wheel's
# version is derived from Version.txt rather than hand-copied into pyproject.toml.
#
#   0.20.0 + "beta.2+38"  ->  0.20.0b2.dev38    a snapshot, sorts BEFORE the release
#   0.20.0 + "beta.2"     ->  0.20.0b2          the release itself, when it is tagged
#   0.20.0 + ""           ->  0.20.0            production
#
# The build counter maps to .devN and NOT .postN, because post-releases sort AFTER:
# publishing a snapshot as plain 0.20.0b2 would permanently take the release's own
# number, forcing the real beta.2 to ship as a post-release of a snapshot. A published
# version can never be reused, so this is not a mistake that can be undone. PyPI also
# rejects local version labels, which is why the +N counter has to move rather than ride
# along. Aborts on an unrecognized stage rather than guessing a version to publish.
function(mila_pep440_version numeric prerelease out_version)
    if(prerelease STREQUAL "")
        set(${out_version} "${numeric}" PARENT_SCOPE)
        return()
    endif()

    if(NOT prerelease MATCHES "^(alpha|beta|rc)\\.([0-9]+)(\\+([0-9]+))?$")
        message(FATAL_ERROR
            "Version.txt prerelease '${prerelease}': expected <alpha|beta|rc>.N[+BUILD]. "
            "Refusing to guess a PEP 440 version for a published artifact.")
    endif()

    set(_stage "${CMAKE_MATCH_1}")
    set(_ordinal "${CMAKE_MATCH_2}")
    set(_build "${CMAKE_MATCH_4}")

    if(_stage STREQUAL "alpha")
        set(_marker "a")
    elseif(_stage STREQUAL "beta")
        set(_marker "b")
    else()
        set(_marker "rc")
    endif()

    if(_build STREQUAL "")
        set(${out_version} "${numeric}${_marker}${_ordinal}" PARENT_SCOPE)
    else()
        set(${out_version} "${numeric}${_marker}${_ordinal}.dev${_build}" PARENT_SCOPE)
    endif()
endfunction()
