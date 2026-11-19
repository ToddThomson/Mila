# Desktop.SearchKeyStore has one candidate file per platform, each keeping the web search key in that operating system's
# credential store (AssistantInterface.md 8): Windows Credential Manager, the macOS keychain, and on Linux the Secret
# Service through libsecret (libsecret-1-dev and pkg-config to build). Included by the session and by its tests, which are
# built without it in a CPU-only build. Sets MILA_SESSION_KEY_STORE, relative to this folder, and
# MILA_SESSION_KEY_STORE_LIBRARIES.
if( WIN32 )
    set( MILA_SESSION_KEY_STORE "Src/Settings/Desktop.SearchKeyStore.Windows.ixx" )
    set( MILA_SESSION_KEY_STORE_LIBRARIES advapi32 )
elseif( APPLE )
    set( MILA_SESSION_KEY_STORE "Src/Settings/Desktop.SearchKeyStore.Apple.ixx" )
    set( MILA_SESSION_KEY_STORE_LIBRARIES "-framework Security" "-framework CoreFoundation" )
else()
    find_package( PkgConfig REQUIRED )
    pkg_check_modules( MILA_LIBSECRET REQUIRED IMPORTED_TARGET libsecret-1 )
    set( MILA_SESSION_KEY_STORE "Src/Settings/Desktop.SearchKeyStore.Linux.ixx" )
    set( MILA_SESSION_KEY_STORE_LIBRARIES PkgConfig::MILA_LIBSECRET )
endif()
