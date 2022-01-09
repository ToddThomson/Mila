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

export module Dnn.ModelBuilder;

import CuDnn.Context;
import CuDnn.Descriptor;
import CuDnn.Error;

using namespace Mila::Dnn::CuDNN;

namespace Mila::Dnn
{
    export class DnnModelBuilder
    {
    public:

        DnnModelBuilder( const ManagedCudnnContext& cudnnContext )
        {
            std::cout << "DnnModelBuilder() constructor\n";
            cudnn_context_ = cudnnContext;
        }

        /// <summary>
        /// Factory for <seealso ref="Descriptor" /> 
        /// </summary>
        /// <typeparam name="TDesc">The type of Descriptor to create.</typeparam>
        /// <returns>The reference to the descriptor created.</returns>
        template<class TDesc>
        TDesc Create()
        {
            return TDesc( cudnn_context_->GetCudnnHandle() );
        }

        const ManagedCudnnContext& GetCudnnContext()
        {
            return cudnn_context_;
        }

        DnnModelBuilder() = default;

    private:
        
        ManagedCudnnContext cudnn_context_ = nullptr;
    };
}