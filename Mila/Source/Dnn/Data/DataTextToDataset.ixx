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

/* Modified code from: https://github.com/jcjohnson/torch-rnn
 
Copyright (c) 2016 Justin Johnson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

module;
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <map>

export module Data.TextToDataset;

import Data.H5DatasetWriter;

namespace fs = std::filesystem;
using namespace Mila::Dnn::Data::H5;

namespace Mila::Dnn::Data
{
    /// <summary>
    /// A class to convert a text file to an H5 formatted dataset.
    /// </summary>
    export class TextToDataset
    {
    public:

        /// <summary>
        /// Creates a dataset from a text file with the specified test, validation
        /// and training splits.
        /// </summary>
        /// <param name="file_path"></param>
        /// <param name="test_split"></param>
        /// <param name="validation_split"></param>
        TextToDataset( const fs::path& file_path, float test_split = 0.1f, float validation_split = 0.1f )
            : file_path_( file_path), test_split_( test_split ), validation_split_( validation_split )
        {
            if ( test_split + validation_split >= 1.0f )
            {
                throw std::invalid_argument( "Invalid dataset splits." );
            }
            
            if ( !std::filesystem::exists( file_path ))
            {
                throw std::invalid_argument( "Input text file does not exist." );
            }
        }
            
        /// <summary>
        /// 
        /// </summary>
        void CreateDataset()
        {
            std::ifstream text_file( file_path_ );

            if ( !text_file.is_open() )
            {
                throw std::invalid_argument( "Cannot open text file." );
            }

            // First go through the file once to determine its size and to 
            // build the vocabulary token map.
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

            std::cout << "File size: " << std::to_string( total_chars ) << std::endl;
            std::cout << "Vocab size: " << std::to_string( tokens.size() ) << std::endl;

            vocabulary_size = tokens.size();
            text_size = total_chars;

            // Now we can figure out the split sizes
            int validation_size = validation_split_ * total_chars;
            int testing_size = test_split_ * total_chars;
            int training_size = total_chars - validation_size - testing_size;

            // Just load data into memory ... we'll have to do something more clever
            // for huge datasets but this should be fine for now

            std::vector<std::vector<char>> splits = {
                std::vector<char>( training_size ),
                std::vector<char>( validation_size ),
                std::vector<char>( testing_size )
            };

            // Go through the file again and write data to our splits vectors
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

            std::string filename_h5 = file_path_.stem().string() + ".h5";

            auto h5Writer = H5DatasetWriter( filename_h5 );

            h5Writer.WriteDataset<char>( "training_ds", splits[ TRAINING_SET ] );
            h5Writer.WriteDataset<char>( "validation_ds", splits[ VALIDATION_SET ] );
            h5Writer.WriteDataset<char>( "testing_ds", splits[ TESTING_SET ] );
            
            // Convert the m
            std::vector<int> vocabulary_vector;

            for ( const auto& [key, value ] : tokens ) {
                vocabulary_vector.push_back( key );
                vocabulary_vector.push_back( value );
            }

            h5Writer.WriteDataset<int>( "vocabulary_ds", vocabulary_vector );
        }

    private:

        const fs::path file_path_;
        
        float validation_split_ = 0.1f;
        float test_split_ = 0.1f;

        size_t vocabulary_size = 0;
        size_t text_size = 0;

        const int TRAINING_SET = 0;
        const int VALIDATION_SET = 1;
        const int TESTING_SET = 2;
    };
}