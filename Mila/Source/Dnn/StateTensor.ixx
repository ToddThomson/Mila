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
#include <vector>
#include <stdexcept>
#include <cudnn.h>

export module Dnn.StateTensor;

import Cuda.Memory;

import CuDnn.Context;
import CuDnn.Descriptor;
import CuDnn.Error;
import CuDnn.Utils;

using namespace Mila::Dnn::CuDNN;

namespace Mila::Dnn
{
    /// <summary>
    /// An n-Dimensional state tensor.
    /// </summary>
    export class StateTensor : public CuDNN::Descriptor
    {
    public:

        StateTensor() = default;

        StateTensor( const ManagedCudnnHandle& handle )
            : Descriptor( handle, CUDNN_TENSOR_DESCRIPTOR )
        {
        }

        /// <summary>
        /// Sets the data type of the Tensor elements.
        /// </summary>
        /// <param name="data_type">Tensor elements data type.</param>
        /// <returns>StateTensor reference</returns>
        auto SetDataType( cudnnDataType_t data_type ) -> StateTensor&
        {
            CheckFinalizedThrow();
            data_type_ = data_type;
            return *this;
        }

        /// <summary>
        /// Sets the tensor state device buffer address. 
        /// </summary>
        /// <param name="address">Buffer address</param>
        /// <returns></returns>
        auto SetAddress( const void* address ) -> StateTensor&
        {
            CheckFinalizedThrow();
            //address_ = address;
            return *this;
        }

        /// <summary>
        /// Sets the number of dimensions and each dimension size of the tensor. 
        /// </summary>
        /// <param name="numberOfDimensions">Numer of dimensions of the Tensor</param>
        /// <param name="dimensionSizes">Array that contain the size of the tensor for every dimension</param>
        /// <returns>Fluent tensor builder</returns>
        auto SetDimensions( int dimensions, int shape[], int strides[]) -> StateTensor&
        {
            CheckFinalizedThrow();
            
            if ( dimensions >= CUDNN_DIM_MAX || dimensions <= 2 )
            {
                throw std::runtime_error( "Tensor dimensions is outside range." );
            }
             
            dimensions_ = dimensions;
            std::copy( shape, shape + dimensions, shape_ );
            std::copy( strides, strides + dimensions, strides_ );

            return *this;
        }

        const int GetDimensions() const
        {
            return dimensions_;
        }

        const void* GetAddress()
        {
            // TJT: FIXME
            return nullptr;// address_;
        }

        std::string ToString() const override
        {
            std::stringstream ss;
            char sep = ' ';
            ss << "StateTensor:: "
                << " Datatype: " << CuDNN::to_string( data_type_ );

            return ss.str();
        }

        cudnnStatus_t Finalize() override
        {
            return Descriptor::SetFinalized();
        }
        
    private:

        cudnnDataType_t data_type_ = CUDNN_DATA_FLOAT;
        int dimensions_ = -1;
        int shape_[ CUDNN_DIM_MAX + 1 ] = { -1 };
        int strides_[ CUDNN_DIM_MAX + 1 ] = { -1 };
        
        //CudaMemory state_;
    };
}