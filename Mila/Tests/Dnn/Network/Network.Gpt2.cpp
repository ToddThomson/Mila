#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <fstream>

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Dnn::Serialization;

namespace Dnn::Networks::Tests
{
    // Minimal concrete Network used for testing importModel (implements required hook).
    // This TestNetwork constructs a tiny GPT-2-like component hierarchy:
    //  - "tf" -> "layer_0" -> "mlp" -> "fc_0"
    // Composite nodes are instances of CompositeComponent and can be traversed
    // with dot-separated paths (matching importModel's path resolution).
    class TestNetwork : public Network<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        explicit TestNetwork( const std::string& name )
            : Network<DeviceType::Cpu, TensorDataType::FP32>( name )
        {
            // Build a minimal component graph (context-independent).
            // Use CompositeComponent for intermediate and leaf nodes so path traversal works.
            using Composite = CompositeComponent<DeviceType::Cpu, TensorDataType::FP32>;
            auto tf = std::make_shared<Composite>( "tf" );
            auto layer0 = std::make_shared<Composite>( "layer_0" );
            auto mlp = std::make_shared<Composite>( "mlp" );
            auto fc0 = std::make_shared<Composite>( "fc_0" );

            // Assemble hierarchy: tf -> layer_0 -> mlp -> fc_0
            mlp->addComponent( fc0 );
            layer0->addComponent( mlp );
            tf->addComponent( layer0 );

            // Add top-level transformer composite to the network
            this->addComponent( tf );

            // Now create and own an execution context, then propagate it to base/network children.
            owned_ctx_ = createExecutionContext( Device::Cpu() );
            this->setExecutionContext( owned_ctx_.get() );
        }

        const ComponentType getType() const override
        {
            return ComponentType::MockComponent;
        }

    protected:
        void save_( ModelArchive& /*archive*/, SerializationMode /*mode*/ ) const override
        {
            // Not needed for these tests.
        }

    private:
        std::unique_ptr<IExecutionContext> owned_ctx_;
    };

    TEST( NetworkGpt2, ImportModel_FileNotFound_Throws )
    {
        TestNetwork net( "test_net_missing" );
        std::filesystem::path missing = std::filesystem::temp_directory_path() / "mila_nonexistent_model.bin";

        // Ensure file does not exist
        if ( std::filesystem::exists( missing ) )
            std::filesystem::remove( missing );

        EXPECT_THROW( net.importModel( missing ), std::runtime_error );
    }

    TEST( NetworkGpt2, ImportModel_Gpt2Small_FileProvided_ImportsOrSkips )
    {
        TestNetwork net( "test_net_gpt2" );

        // Prefer the working-directory-relative path first (common for CI),
        // then search upward from the source file location for a repository copy.
        std::filesystem::path repo_relative = std::filesystem::path( "." ) / "data" / "Weights" / "Gpt2" / "gpt2_small.bin";

        if ( !std::filesystem::exists( repo_relative ) )
        {
            // Walk upward from this source file to find the repository root that contains a Data/Weights folder.
            std::filesystem::path source = std::filesystem::path(__FILE__);
            auto dir = source.parent_path(); // .../Tests/Dnn/Network

            while ( !dir.empty() )
            {
                auto candidate = dir / ".." / "data" / "Weights" / "Gpt2" / "gpt2_small.bin";
                candidate = std::filesystem::weakly_canonical(candidate);

                if ( std::filesystem::exists( candidate ) )
                {
                    repo_relative = candidate;
                    break;
                }

                dir = dir.parent_path();
            }
        }

        if ( !std::filesystem::exists( repo_relative ) )
        {
            GTEST_SKIP() << "Gpt2 weights not present at " << repo_relative.string() << " - skipping import test.";
        }

        // importModel should succeed for the provided GPT-2 weights file.
        EXPECT_NO_THROW( net.importModel( repo_relative ) );
    }
}