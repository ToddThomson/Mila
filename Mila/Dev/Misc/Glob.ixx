// TJT: This code will be deprecated. It is currently used by the Gpt2 Sample which will be refactored soon.

module;
#include <string>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <time.h>
#include <cstring>

export module Misc.Glob;

namespace Mila::Misc
{
    export struct glob_t {
        size_t gl_pathc;    // Count of matched pathnames
        char** gl_pathv;    // List of matched pathnames
    };

    export void globfree( glob_t* pglob ) {
        for ( size_t i = 0; i < pglob->gl_pathc; ++i ) {
            free( pglob->gl_pathv[ i ] );
        }
        free( pglob->gl_pathv );
    }

    // Convert glob pattern to regex pattern
    inline std::string glob_to_regex( const std::string& pattern ) {
        std::string result;
        for ( char c : pattern ) {
            if ( c == '*' ) result += ".*";
            else if ( c == '?' ) result += ".";
            else if ( c == '.' || c == '+' || c == '(' || c == ')' ||
                c == '[' || c == ']' || c == '^' || c == '$' ) {
                result += '\\';
                result += c;
            }
            else result += c;
        }
        return result;
    }

    export int glob( const char* pattern, int ignored_flags, int (*ignored_errfunc)(const char* epath, int eerrno), glob_t* pglob ) {
        std::filesystem::path fs_pattern( pattern );
        std::filesystem::path directory = fs_pattern.parent_path();
        std::string file_pattern = fs_pattern.filename().string();

        // If directory is empty, use current directory
        if ( directory.empty() ) directory = ".";

        // Convert glob pattern to regex
        std::regex file_regex( glob_to_regex( file_pattern ) );

        size_t file_count = 0;
        size_t max_files = 64000;

        pglob->gl_pathv = (char**)malloc( max_files * sizeof( char* ) );
        if ( pglob->gl_pathv == NULL ) {
            return 2; // Memory allocation failed
        }

        // Check if directory exists
        if ( !std::filesystem::exists( directory ) || !std::filesystem::is_directory( directory ) ) {
            free( pglob->gl_pathv );
            return 1; // Directory not found
        }

        try {
            for ( const auto& entry : std::filesystem::directory_iterator( directory ) ) {
                if ( file_count >= max_files ) break;

                std::string filename = entry.path().filename().string();
                if ( std::regex_match( filename, file_regex ) ) {
                    std::string full_path = entry.path().string();
                    pglob->gl_pathv[ file_count ] = strdup( full_path.c_str() );
                    if ( pglob->gl_pathv[ file_count ] == NULL ) {
                        return 2; // Memory allocation for filename failed
                    }
                    file_count++;
                }
            }
        }
        catch ( const std::filesystem::filesystem_error& ) {
            return 1; // Error accessing directory
        }

        pglob->gl_pathc = file_count;
        return 0;
    }
}
