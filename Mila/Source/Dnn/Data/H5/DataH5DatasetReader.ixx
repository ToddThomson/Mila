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
#include <vector>
#include "H5Cpp.h"

export module Data.H5DatasetReader;

import Data.H5DataTypeMapper;

namespace fs = std::filesystem;
using namespace H5;

export namespace Mila::Dnn::Data::H5
{
    export class H5DatasetReader
    {
    public:

        H5DatasetReader( const std::string& filename )
        {
            h5_file_ = H5File( filename, H5F_ACC_RDONLY );
        }

        template <class TElement>
        int ReadDataset( const std::string& datasetName, std::vector<TElement>& data )
        {
            try
            {
                DataSet dataset = h5_file_.openDataSet( datasetName );
                
                DataSpace filespace = dataset.getSpace();

                int rank = filespace.getSimpleExtentNdims();

                hsize_t dims[ 1 ];
                int data_size = filespace.getSimpleExtentDims( dims );

                std::cout << "dataset rank = " << rank << ", number of elements: " << (unsigned long)(dims[ 0 ]) << " x " << std::endl;

                data.reserve( (unsigned long)(dims[ 0 ]) );
                data.resize( (unsigned long)(dims[ 0 ]) );
                DataSpace mspace1( 1, dims );

                dataset.read( data.data(), get_data_type<TElement>(), mspace1, filespace);
                //data.resize( (unsigned long)(dims[ 0 ]) );

            } // end of try block

            // catch failure caused by the H5File operations
            catch ( FileIException error )
            {
                error.printErrorStack();
                return -1;
            }

            // catch failure caused by the DataSet operations
            catch ( DataSetIException error )
            {
                error.printErrorStack();
                return -1;
            }

            // catch failure caused by the DataSpace operations
            catch ( DataSpaceIException error )
            {
                error.printErrorStack();
                return -1;
            }

            // catch failure caused by the DataSpace operations
            catch ( DataTypeIException error )
            {
                error.printErrorStack();
                return -1;
            }

            return 0; // successfully terminated            
        }

    private:

        H5File h5_file_;
    };

    template int H5DatasetReader::ReadDataset<char>( const std::string& datasetName, std::vector<char>& data );
    template int H5DatasetReader::ReadDataset<int>( const std::string& datasetName, std::vector<int>& data );
}