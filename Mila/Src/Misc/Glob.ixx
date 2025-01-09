/*
 * Copyright 2024..25 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the Mila end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

module;

#include <string>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <time.h>

export module Misc.Glob;

namespace Mila::Misc
{
    export struct glob_t {
        size_t gl_pathc;    // Count of matched pathnames
        char** gl_pathv;    // List of matched pathnames
    };

    inline void replace_forward_slashes( char* str ) {
        while ( *str ) {
            if ( *str == '/' ) {
                *str = '\\';
            }
            str++;
        }
    }

    export void globfree( glob_t* pglob ) {
        for ( size_t i = 0; i < pglob->gl_pathc; ++i ) {
            free( pglob->gl_pathv[ i ] ); // Free the allocated memory for each filename
        }
        free( pglob->gl_pathv ); // Free the allocated memory for the list of filenames
    }

    export int glob( const char* pattern, int ignored_flags, int (*ignored_errfunc)(const char* epath, int eerrno), glob_t* pglob ) {
        struct _finddata_t find_file_data;
        char full_path[ 576 ]; // stored in pglob->gl_pathv[n]
        char directory_path[ 512 ] = { 0 }; // Store the directory path from the pattern
        char pattern_copy[ 512 ]; // Copy of the pattern to modify

        strncpy_s( pattern_copy, sizeof( pattern_copy ) - 1, pattern, sizeof( pattern_copy ) - 1 );

        replace_forward_slashes( pattern_copy ); // Replace forward slashes with backslashes

        if ( strchr( pattern_copy, '\\' ) != (void*)NULL ) {
            strncpy_s( directory_path, sizeof( directory_path ) - 1, pattern_copy, strrchr( pattern_copy, '\\' ) - pattern_copy + 1 );
            directory_path[ strrchr( pattern_copy, '\\' ) - pattern_copy + 1 ] = '\0';
        }

        // find the first file matching the pattern in the directory
        intptr_t find_handle = _findfirst( pattern_copy, &find_file_data );

        if ( find_handle == -1 ) {
            return 1; // No files found
        }

        size_t file_count = 0;
        size_t max_files = 64000; // hard-coded limit for the number of files

        pglob->gl_pathv = (char**)malloc( max_files * sizeof( char* ) ); // freed in globfree

        if ( pglob->gl_pathv == NULL ) {
            _findclose( find_handle );
            return 2; // Memory allocation failed
        }

        do {
            if ( file_count >= max_files ) {
                _findclose( find_handle );
                return 2; // Too many files found
            }

            snprintf( full_path, sizeof( full_path ), "%s%s", directory_path, find_file_data.name );

            pglob->gl_pathv[ file_count ] = _strdup( full_path ); // freed in globfree

            if ( pglob->gl_pathv[ file_count ] == NULL ) {
                _findclose( find_handle );
                return 2; // Memory allocation for filename failed
            }
            file_count++;
        } while ( _findnext( find_handle, &find_file_data ) == 0 );

        _findclose( find_handle );

        pglob->gl_pathc = file_count;
        return 0;
    }
}