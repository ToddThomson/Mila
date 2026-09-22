# Mila version helpers.

# mila_read_version(<version_file> <out_numeric> <out_prerelease>)
#
# Reads a SemVer string (X.Y.Z or X.Y.Z-PRERELEASE, e.g. "0.21.0-dev+7") from
# version_file and returns, via the named output variables:
#   out_numeric    -- the numeric major.minor.patch triple, suitable for
#                     project(VERSION ...).
#   out_prerelease -- the optional prerelease label (e.g. "dev+7"), empty if
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
#   0.21.0 + "dev+7"  ->  0.21.0.dev7    a snapshot, sorts BEFORE the release
#   0.21.0 + ""       ->  0.21.0         the release itself, when it is tagged
#
# The build counter maps to .devN and NOT .postN, because post-releases sort AFTER:
# publishing a snapshot as plain 0.21.0 would permanently take the release's own number,
# forcing the real release to ship as a post-release of a snapshot. A published version
# can never be reused, so this is not a mistake that can be undone. PyPI also rejects
# local version labels, which is why the +N counter has to move rather than ride along.
#
# One stage named `dev` with no ordinal, since 0.21.0. The alpha/beta/rc ladder it
# replaced needed both an ordinal and a counter and had only one PEP 440 slot to put them
# in, which is how 0.20.0-beta.2+38 became the crowded 0.20.0b2.dev38. The counter is
# REQUIRED here rather than optional: the ordinal used to be what distinguished two
# pre-release trees, so with it gone the counter is the only thing that does, and a bare
# `dev` names no version worth publishing. Aborts rather than guessing one.
#
# Only a MINOR is published to PyPI, so in practice this runs for a minor's snapshots and
# for the minor itself. A patch is a git tag for source consumers and builds no wheel --
# it still passes through here, because CMakeLists.txt reads the version unconditionally.
function(mila_pep440_version numeric prerelease out_version)
    if(prerelease STREQUAL "")
        set(${out_version} "${numeric}" PARENT_SCOPE)
        return()
    endif()

    if(NOT prerelease MATCHES "^dev\\+([0-9]+)$")
        message(FATAL_ERROR
            "Version.txt prerelease '${prerelease}': expected dev+N. The alpha/beta/rc "
            "ladder was retired when the 0.21.0 cycle opened; there is one stage, and a "
            "dev build is never tagged. Refusing to guess a PEP 440 version for a "
            "published artifact.")
    endif()

    set(${out_version} "${numeric}.dev${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()
