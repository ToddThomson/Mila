#include "gtest/gtest.h"
#include "Mila.h"
#include "Common.h"

using namespace Mila::Dnn;

namespace Mila::Tests::Dnn
{
    class DropoutTestModel : public DnnModel
    {
    public:

        DropoutTestModel() : DnnModel( DnnModelOptions() )
        {
        }

        void OnModelBuilding( DnnModelBuilder builder ) override
        {
            auto options = GetDefaultRnnOptions();

            dropout_ = builder.Create<Dropout>();

            dropout_.SetProbability( 0.1f )
                .SetSeed( 739134ull );

            dropout_.Finalize();
        }

        Dropout& GetDropout()
        {
            return dropout_;
        }

    private:

        Dropout dropout_;
    };
    
    TEST( Dnn_Operations, Create_Dropout )
    {
        DropoutTestModel app = DropoutTestModel();

        app.BuildModel();

        Dropout& dropout = app.GetDropout();
        auto status = dropout.get_status();

        std::cout << dropout.ToString() << std::endl;

        if ( status != CUDNN_STATUS_SUCCESS )
        {
            std::cout << std::endl << "Test Failed!" << std::endl
                << "Status: " << to_string( status ) << std::endl
                << "Error: " << dropout.get_error();
        }

        EXPECT_TRUE( status == CUDNN_STATUS_SUCCESS );
    };
}