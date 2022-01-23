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
#include <string>
#include <filesystem>
#include <stdexcept>
#include <map>
#include <vector>

export module Data.Dataset;

import Data.DatasetType;
import Data.H5DatasetReader;

namespace fs = std::filesystem;

namespace Mila::Dnn::Data
{
    /// <summary>
    /// Represents a block of (X,Y) input, output character vectors.
    /// </summary>
    export using XYPair = std::pair<std::vector<char>, std::vector<char>>;

    export class Dataset
    {
    public:

        Dataset( const fs::path& datasetPath, int batchSize, int sequenceLength )
            : path_( datasetPath ), batch_size_( batchSize ), sequence_length_( sequenceLength )
        {
            if ( !fs::exists( path_ ) )
            {
                throw std::invalid_argument( "Dataset file does not exist." );
            }

            block_size_ = batch_size_ * sequence_length_;

            ReadVocabulary();
        }

        /// <summary>
        /// Loads the specified dataset type.
        /// </summary>
        /// <param name="datasetType"></param>
        void Load( const DatasetType datasetType )
        {
            H5::H5DatasetReader ds_reader = H5::H5DatasetReader( path_.string() );

            ds_reader.ReadDataset<char>( to_string( datasetType ), dataset_ );

            int dataset_size = dataset_.size();

            max_blocks_ = (dataset_size - 1) / block_size_;
        }

        const std::map<int,int>& GetVocabulary()
        {
            return vocabulary_;
        }

        /// <summary>
        /// Gets the next block from the dataset.
        /// </summary>
        /// <returns>X,Y input pair</returns>
        XYPair NextBlock()
        {
            if ( EndOfDataset() )
            {
                throw std::out_of_range( "Error reading past end of dataset." );
            }

            std::vector<char> x(
                dataset_.begin() + dataset_offset_,
                dataset_.begin() + dataset_offset_ + block_size_ );

            std::vector<char> y(
                dataset_.begin() + dataset_offset_ + 1,
                dataset_.begin() + dataset_offset_ + 1 + block_size_ );

            dataset_offset_ += block_size_;
            block_index_++;

            return XYPair( x, y );
        }

        bool EndOfDataset()
        {
            return block_index_ >= max_blocks_;
        }

        void Reset()
        {
            block_index_ = 0;
        }

        int BlockCount()
        {
            return max_blocks_;
        }

    private:

        void ReadVocabulary()
        {
            H5::H5DatasetReader ds_reader = H5::H5DatasetReader( path_.string() );

            std::vector<int> vocabulary_vector;
            ds_reader.ReadDataset<int>( to_string( DatasetType::vocabulary ), vocabulary_vector );

            for ( auto it = vocabulary_vector.begin(); it != vocabulary_vector.end(); it++ ) {
                vocabulary_[ *it++ ] = *it;
            }
        }

    private:

        fs::path path_;

        int batch_size_;
        int sequence_length_;

        int dataset_offset_ = 0;

        int block_index_ = 0;
        int max_blocks_ = 0;
        int block_size_ = 0;

        std::map<int, int> vocabulary_;
        std::vector<char> dataset_;
    };
}