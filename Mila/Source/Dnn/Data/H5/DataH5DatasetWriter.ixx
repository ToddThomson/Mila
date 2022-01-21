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
#include "H5Cpp.h"

export module Data.H5DatasetWriter;

import Data.H5DataTypeMapper;

namespace fs = std::filesystem;
using namespace H5;

namespace Mila::Dnn::Data::H5
{
    /*template<typename TElement>
    const PredType& get_data_type();

    template<>
    const PredType& get_data_type<float>() { return PredType::NATIVE_FLOAT; }

    template<>
    const PredType& get_data_type<int>() { return PredType::NATIVE_INT; }

    template<>
    const PredType& get_data_type<char>() { return PredType::NATIVE_CHAR; }*/

    export class H5DatasetWriter
    {
    public:

        H5DatasetWriter( std::string filename )
        {
            h5_file_ = H5File( filename, H5F_ACC_TRUNC );
        }

        template <class TElement>
        int WriteDataset( const std::string& datasetName, const std::vector<TElement>& data )
        {
            try
            {
                hsize_t splits_dimsf[ 1 ]{ data.size()};
                //splits_dimsf[ 0 ] = data.size();
                DataSpace dataspace( 1, splits_dimsf );

                //PredType predType = get_data_type<TElement>();
                IntType datatype( get_data_type<TElement>() ); // PredType::NATIVE_UCHAR ); 
                datatype.setOrder( H5T_ORDER_LE );

                /*
                 * Create a new dataset within the file using defined dataspace and
                 * datatype and default dataset creation properties.
                 */
                DataSet dataset = h5_file_.createDataSet(
                    datasetName, datatype, dataspace );

                /*
                 * Write the data to the dataset using default memory space, file
                 * space, and transfer properties.
                 */
                dataset.write( data.data(), get_data_type<TElement>() ); // PredType::NATIVE_UCHAR );

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
 
    template int H5DatasetWriter::WriteDataset<char>( const std::string& datasetName, const std::vector<char>& data );
    template int H5DatasetWriter::WriteDataset<int>( const std::string& datasetName, const std::vector<int>& data );
}