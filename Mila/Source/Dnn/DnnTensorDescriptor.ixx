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
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cudnn.h>

export module Dnn.TensorDescriptor;

import CuDnn.Descriptor;
import CuDnn.Error;
import CuDnn.Utils;

using namespace Mila::Dnn::CuDnn;

namespace Mila::Dnn
{
    /// <summary>
    /// A generic n-Dimensional Tensor.
    /// Use <seealso member="DnnModelBuilder"/> to build an instance of this class.
    /// </summary>
    export class Tensor : public CuDnn::Descriptor
    {
        friend class DnnModelBuilder;

    public:

        Tensor() = default;

        Tensor( ManagedCudnnHandle& handle )
            : Descriptor( handle, CUDNN_TENSOR_DESCRIPTOR )
        {
            std::cout << "Tensor()" << std::endl;
        }

        std::string ToString() const override
        {
            std::stringstream ss;
            char sep = ' ';
            ss << "Tensor:: " << std::endl
                << "Datatype: " << CuDnn::to_string( data_type_ ) << std::endl
                << "Dimensions: " << std::to_string( dimensions_ ) << std::endl;

            return ss.str();
        }

        /// <summary>
        /// Sets the data type of the Tensor elements.
        /// </summary>
        /// <param name="data_type"></param>
        /// <returns>Tensor builder</returns>
        auto SetDataType( cudnnDataType_t data_type ) -> Tensor&
        {
            data_type_ = data_type;
            return *this;
        }

        /// <summary>
        /// Sets the number of dimensions and each dimension size of the tensor. 
        /// </summary>
        /// <param name="numberOfDimensions">Numer of dimensions of the Tensor</param>
        /// <param name="dimensionSizes">Array that contain the size of the tensor for every dimension</param>
        /// <returns>Fluent tensor builder</returns>
        auto SetDimensions( int dimensions, const std::vector<int>& shape, const std::vector<int>& strides ) -> Tensor&
        {
            if ( dimensions >= CUDNN_DIM_MAX || dimensions <= 2 )
            {
                throw std::runtime_error( "Tensor dimensions is outside range." );
            }

            dimensions_ = dimensions;
            shape_.assign( shape.begin(), shape.end() );
            strides_.assign( strides.begin(), strides.end() );

            return *this;
        }

        const int GetDimensions() const
        {
            return dimensions_;
        }

        cudnnStatus_t Finalize() override
        {
            auto status = cudnnSetTensorNdDescriptor(
                static_cast<cudnnTensorDescriptor_t>(GetOpaqueDescriptor()),
                data_type_,
                dimensions_,
                shape_.data(),
                strides_.data() );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                SetErrorAndThrow(
                    /* this, */ status, "Failed to set Tensor descriptor.");

                return status;
            }

            return Descriptor::SetFinalized();
        }

    private:

        // Tensor Properties
        cudnnDataType_t data_type_ = CUDNN_DATA_FLOAT; 
        int dimensions_ = -1;
        std::vector<int> shape_ = { -1,-1,-1,-1,-1,-1 ,-1,-1,-1 };
        std::vector<int> strides_ ={ -1,-1,-1,-1,-1,-1 ,-1,-1,-1 };
    };
}