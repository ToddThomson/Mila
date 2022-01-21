#include "gtest/gtest.h"
#include "../Common.h"
#include <iostream>

import Dnn.Model;
import Dnn.RnnOpDescriptor;
import Dnn.DropoutDescriptor;
import CuDnn.Utils;

namespace Mila::Tests::Dnn
{
    class RnnOpTestModel : public Mila::Dnn::DnnModel
    {
    public:

        RnnOpTestModel() : DnnModel( DnnModelOptions() )
        {
            options_ = GetDefaultRnnOptions();
        }

        void OnModelBuilding( const DnnModelBuilder& builder ) override
        {
            CreateDropout( builder );

            rnnOp_ = builder.Create<Mila::Dnn::RnnOperation>();

            rnnOp_.SetAlgorithm( options_.algorithm )
                .SetCellMode( options_.cellMode )
                .SetBiasMode( options_.biasMode )
                .SetDirectionMode( options_.dirMode )
                .SetInputMode( options_.inputMode )
                .SetDataType( options_.dataType )
                .SetMathPrecision( options_.mathPrecision )
                .SetMathType( options_.mathType )
                .SetInputSize( options_.inputSize )
                .SetHiddenSize( options_.hiddenSize )
                .SetProjSize( options_.projSize )
                .SetNumLayers( options_.numLayers )
                .SetDropout( dropout_ );

            auto status = rnnOp_.Finalize();
        }

        void CreateDropout( DnnModelBuilder builder )
        {
            dropout_ = builder.Create<Dropout>();

            dropout_.SetProbability( 0.1f )
                .SetSeed( 739134ull );

            dropout_.Finalize();
        }

        const RnnOperation& GetRnnOp() const
        {
            return rnnOp_;
        }

        /*const Dropout& GetDropout()
        {
            return dropout_;
        }*/

    private:

        RnnModelOptions options_;

        /// <summary>
        /// The RNN operation.
        /// </summary>
        RnnOperation rnnOp_;

        /// <summary>
        /// Dropout operation required for RNN operation.
        /// </summary>
        Dropout dropout_;
    };

    TEST( Dnn_Operations, Create_RnnOp_Float )
    {
        auto app = RnnOpTestModel();

        app.BuildModel();

        auto status = app.GetRnnOp().get_status();
        std::cout << app.GetRnnOp().ToString() << std::endl;

        if ( status == CUDNN_STATUS_SUCCESS )
        {
            std::cout << "Test passed successfully.\n";
        }
        else
        {
            std::cout << "Test Failed with Status: " << Mila::Dnn::CuDnn::to_string( status ) << std::endl;
        }

        EXPECT_TRUE( status == CUDNN_STATUS_SUCCESS );
    }
}