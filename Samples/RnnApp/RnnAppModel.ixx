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

export module RnnApp.Model;

import Dnn.RnnModel;
import CuDnn.Utils;

namespace RnnApp::Model
{
    export void TrainModel()
    {
        auto model = Mila::Dnn::RnnModel<float>();

        model.BuildModel();
        model.Train();

        std::cout << "RnnModel::rnnOp member:" << std::endl
            << model.GetRnnOp().ToString() << std::endl;

        auto status = model.GetRnnOp().get_status();
        auto error = model.GetRnnOp().get_error();

        if ( status == CUDNN_STATUS_SUCCESS )
        {
            std::cout << std::endl << "Test passed successfully." << std::endl;
        }
        else
        {
            std::cout << std::endl << "Test Failed!" << std::endl
                << "Status: " << Mila::Dnn::CuDNN::to_string( status ) << std::endl
                << "Error: " << error << std::endl;
        }
    };

    //export class RnnAppModel : Mila::Dnn::RnnModel<float>
    //{

    //};
}