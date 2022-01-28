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
#include <iostream>
#include <sstream>
#include <cudnn.h>

export module Dnn.DropoutDescriptor;

import Cuda.Memory;
import Cuda.Error;

import CuDnn.Context;
import CuDnn.Descriptor;
import CuDnn.Error;

using namespace Mila::Dnn::Cuda;
using namespace Mila::Dnn::CuDnn;

namespace Mila::Dnn
{
    export class Dropout : public Descriptor
    {
        friend class DnnModelBuilder;

    public:

        Dropout() = default;
        
        Dropout( const ManagedCudnnHandle& cudnnHandle )
            : Descriptor( cudnnHandle, CUDNN_DROPOUT_DESCRIPTOR )
        {
            std::cout << "Dropout()\n";
        }

       /// <summary>
       /// Sets the probability with which the value from input is set to zero during the dropout layer.
       /// </summary>
       /// <param name="dropout">probability</param>
       /// <returns>Dropout object</returns>
        auto SetProbability( float probability ) -> Dropout&
        {
            CheckFinalizedThrow();

            probability_ = probability;
            
            return *this;
        }

        /// <summary>
        /// Sets the seed used to initialize random number generator states
        /// </summary>
        /// <param name="seed"></param>
        /// <returns></returns>
        auto SetSeed( unsigned long long seed ) -> Dropout&
        {
            if ( !IsFinalized() )
            {
                seed_ = seed;
            }
            
            return *this;
        };

        /// <summary>
        /// 
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        cudnnStatus_t Finalize() override
        {
            // TJT: TODO Property validation

            size_t stateSize = 0;

            auto status = cudnnDropoutGetStatesSize(
                cudnn_handle_->GetOpaqueHandle(),
                &stateSize );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                SetErrorAndThrow(
                    /* this, */ status, "Failed to get Dropout States Size.");

                return status;
            }

            state_memory_ = CudaMemory( stateSize );

            status = cudnnSetDropoutDescriptor(
                static_cast<cudnnDropoutDescriptor_t>(GetOpaqueDescriptor()),
                cudnn_handle_->GetOpaqueHandle(),
                probability_,
                state_memory_.GetBuffer(),
                state_memory_.GetBufferSize(),
                seed_ );

            if ( status != CUDNN_STATUS_SUCCESS )
            {
                SetErrorAndThrow(
                    /* this, */ status, "Failed to create Dropout descriptor.");

                return status;
            }

            return Descriptor::SetFinalized();
        }

        std::string ToString() const override
        {
            std::stringstream ss;
            char sep = ' ';
            ss << "Dropout:: " << std::endl
                << " Probability: " << probability_ << std::endl
                << " Seed: " << seed_ << std::endl;

            return ss.str();
        }

        Dropout( Dropout&& other ) : Descriptor( std::move( other ) )
        {
            std::cout << "Dropout move constructor" << std::endl;
            other.state_memory_ = std::move( other.state_memory_ );
        }

        Dropout& operator=( Dropout&& other )
        {
            // TJT: Do we need to deep copy here?
            std::cout << "Dropout::move assignment" << std::endl;

            Descriptor::operator=( std::move( other ) );

            state_memory_ = std::move( other.state_memory_ );
            probability_ = other.probability_;
            seed_ = other.seed_;

            return *this;
        }


    private:

        CudaMemory state_memory_;

        /// <summary>
        /// The probability with which the value from input is set to zero during the dropout layer.
        /// </summary>
        float probability_ = 0.0f;

        /// <summary>
        /// Seed used to initialize random number generator states.
        /// </summary>
        unsigned long long seed_ = 135531ull;
    };
}