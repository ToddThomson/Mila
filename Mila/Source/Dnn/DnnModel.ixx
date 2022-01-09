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
#include <cudnn.h>

export module Dnn.Model;

import CuDnn.Context;
import Dnn.ModelOptions;
import Dnn.ModelBuilder;
import Dnn.NeuralNetType;

using namespace Mila::Dnn::CuDNN;

namespace Mila::Dnn
{
    export class DnnModel
    {
    public:

        DnnModel() : DnnModel( DnnModelOptions() )
        {}

        DnnModel( const DnnModelOptions& options )
        {
            options_ = options;
            context_ = std::make_unique<CudnnContext>();
            builder_ = DnnModelBuilder( context_ );
        }

        const DnnModelBuilder& GetModelBuilder()
        {
            return builder_;
        }

        virtual void BuildModel() final
        {
            OnModelBuilding( builder_ );
        }

    protected:

        /// <summary>
        /// Override to configure the application dnn model.
        /// </summary>
        /// <param name="builder"></param>
        virtual void OnModelBuilding( DnnModelBuilder builder )
        {
        }

    private:

        DnnModel( DnnModel const& ) = delete;
        DnnModel& operator=( DnnModel const& ) = delete;

        DnnModelOptions options_;
        DnnModelBuilder builder_;
        ManagedCudnnContext context_ = nullptr;
       
        neuralNetType_t type_ = RECURRENT_NN_TYPE;
    };
}