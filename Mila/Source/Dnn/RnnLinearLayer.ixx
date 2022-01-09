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
#include <memory>
#include <cudnn.h>

export module Dnn.RnnLinearLayer;

import Dnn.StateTensor;

namespace Mila::Dnn
{
    export class RnnLinearLayer
    {
    public:

        RnnLinearLayer( 
            int layerId, 
            int linearLayerId, 
            StateTensor& weightMatrix, 
            StateTensor& biasVector )
        {
            layer_id_ = layerId;
            linear_layer_id_ = linearLayerId;

            weight_matrix_ = std::move( weightMatrix );
            bias_vector_ = std::move( biasVector );
        }

        std::string ToString()
        {
            std::stringstream ss;
            char sep = ' ';
            ss << "RnnLinearLayer:: " << std::endl;

            return ss.str();
        }

        bool HasWeightMatrix()
        {
            return false;// (weight_matrix_.GetAddress() != NULL);
        }

        bool HasBiasVector()
        {
            return false;// (bias_vector_.GetAddress() != NULL);
        }

    private:

        int layer_id_ = 0;
        int linear_layer_id_ = 0;

        StateTensor weight_matrix_;
        StateTensor bias_vector_;
    };
}