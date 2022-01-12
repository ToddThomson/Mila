/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
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
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <map>

export module Dnn.Data.TextToH5;

import Dnn.Data.H5DatasetWriter;

namespace fs = std::filesystem;
using namespace Mila::Dnn::Data::H5;

namespace Mila::Dnn::Data
{
    export class TextToH5
    {
    public:

        int Convert( fs::path filename )
        {
            std::ifstream text_file( filename );

            if ( !text_file.is_open() )
            {
                std::cout << "failed to open " << filename << std::endl;
                return -1;
            }

            // First go through the file once to see how big it is and to build the vocab
            std::map<char, int> tokens = {};
            int total_chars = 0;
            int index = 1;
            std::string line;

            while ( std::getline( text_file, line ) )
            {
                total_chars += line.length();

                for ( char c : line )
                {
                    auto it = tokens.find( c );
                    if ( it == tokens.end() )
                    {
                        tokens.insert( {c, index++} );
                    }
                }
            }

            std::cout << "Char size: " << std::to_string( total_chars ) << std::endl;
            std::cout << "Vocab size: " << std::to_string( tokens.size() ) << std::endl;

            // Now we can figure out the split sizes
            int validation_size = validation_split_ * total_chars;
            int testing_size = test_split_ * total_chars;
            int training_size = total_chars - validation_size - testing_size;

            // Just load data into memory ... we'll have to do something more clever
            // for huge datasets but this should be fine for now

            //std::vector<char> training_set( train_size );
            //std::vector<char> validation_set( validation_size );
            //std::vector<char> testing_set( test_size );

            std::vector<std::vector<char>> splits = {
                std::vector<char>( training_size ),
                std::vector<char>( validation_size ),
                std::vector<char>( testing_size )
            };

            // Go through the file again and write data to our splits arrays
            text_file.clear();
            text_file.seekg( 0 );

            int split_index = 0;
            int split_position = 0;

            while ( std::getline( text_file, line ) )
            {
                total_chars += line.length();

                for ( char& c : line )
                {
                    splits[ split_index ][ split_position++ ] = tokens[ c ];

                    if ( split_position >= splits[ split_index ].size() )
                    {
                        split_index += 1;
                        split_position = 0;
                    }
                }
            }

            text_file.close();

            std::string filename_h5 = filename.stem().string() + ".h5";

            auto h5Writer = H5DatasetWriter( filename_h5 );

            h5Writer.WriteDataset( "training_ds", splits[ TRAINING_SET ] );
            h5Writer.WriteDataset( "validation_ds", splits[ VALIDATION_SET ] );
            h5Writer.WriteDataset( "testing_ds", splits[ TESTING_SET ] );

            return 0;
        }

    private:

        float validation_split_ = 0.1f;
        float test_split_ = 0.1f;

        const int TRAINING_SET = 0;
        const int VALIDATION_SET = 1;
        const int TESTING_SET = 2;
    };
}