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

export module Data.DatasetLoader;

import Data.H5DatasetReader;
import Data.DatasetType;

namespace fs = std::filesystem;

namespace Mila::Dnn::Data
{
    export using XYPair = std::pair<std::vector<char>, std::vector<char>>;

    export class DatasetLoader
    {
    public:

        /// <summary>
        /// DatasetLoader constructor
        /// </summary>
        /// <param name="datasetType"></param>
        /// <param name="datasetPath"></param>
        /// <param name="batchSize"></param>
        /// <param name="sequenceLength"></param>
        DatasetLoader( const DatasetType datasetType, fs::path datasetPath, int batchSize, int sequenceLength )
        { 
            dataset_path_ = datasetPath;
            batch_size_ = batchSize;
            sequence_length_ = sequenceLength;

            block_size_ = batchSize * sequenceLength;

            ReadDataset( datasetType );
        }

        XYPair Next()
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

        int ReadDataset( const DatasetType datasetType )
        {
            H5::H5DatasetReader ds_reader = H5::H5DatasetReader( dataset_path_.string() );

            ds_reader.ReadDataset( to_string( datasetType ), dataset_ );

            int dataset_size = dataset_.size();
             
            max_blocks_ = (dataset_size - 1) / block_size_;

            return 0;
        }

    private:

        fs::path dataset_path_;

        int batch_size_;
        int sequence_length_;

        int dataset_offset_ = 0;

        int block_index_ = 0;
        int max_blocks_ = 0;
        int block_size_ = 0;

        std::vector<char> dataset_;
    };
}