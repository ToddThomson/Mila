/**
 * @file Component.Cuda.cpp
 * @brief Device companion to Component.cpp: the parameter-staging serialization path.
 *
 * saveParameterToArchive() has two branches. The host-accessible one is covered in
 * Component.cpp; this file covers the device one, which is where the interesting
 * constraint lives. Every reduced precision Mila trains or serves in -- BF16, FP16,
 * FP8, FP4 -- is is_device_only, so isValidTensor rejects a CpuMemoryResource
 * staging buffer at the parameter's own dtype and the helper stages through pinned
 * memory instead. A BF16 parameter is therefore the only way to exercise the branch
 * at all, and it needs a device.
 *
 * GPU-local: compiled only under MILA_ENABLE_CUDA, skipped when no device present.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Core
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        // A leaf owning a single BF16 device parameter -- the minimum needed to drive
        // saveParameterToArchive() down its staging branch.
        class CudaParameterComponent : public Component<DeviceType::Cuda, TensorDataType::BF16>
        {
        public:
            using Base = Component<DeviceType::Cuda, TensorDataType::BF16>;
            using TensorType = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

            CudaParameterComponent( const std::string& name, DeviceId device_id, const shape_t& shape )
                : Base( name ),
                device_id_( device_id ),
                weight_( std::make_shared<TensorType>( device_id, shape ) )
            {}

            TensorType& weight()
            {
                return *weight_;
            }

            void exposeSetExecutionContext( IExecutionContext* context )
            {
                this->setExecutionContext( context );
            }

            void synchronize() override
            {
                this->getExecutionContext()->synchronize();
            }

            dim_t parameterCount() const override
            {
                return weight_->size();
            }

            std::vector<std::string> getParameterNames() const override
            {
                return { "weight" };
            }

            std::vector<ITensor*> getParameters() const override
            {
                return { weight_.get() };
            }

            std::vector<ITensor*> getGradients() const override
            {
                return {};
            }

            void save_( ModelArchive& archive, SerializationMode ) const override
            {
                this->saveParameterToArchive( archive, "weight", *weight_ );
            }

            // Qualified: Mila::Dnn::MemoryStats and Mila::Dnn::Compute::MemoryStats are
            // distinct exported types and both using-directives are in scope here.
            Mila::Dnn::MemoryStats getMemoryStats() const override
            {
                return {};
            }

            const ComponentType getType() const override
            {
                return ComponentType::MockComponent;
            }

            DeviceId getDeviceId() const override
            {
                return device_id_;
            }

            std::string toString() const override
            {
                return std::string( "CudaParameterComponent:" ) + this->getName();
            }

        private:
            DeviceId device_id_;
            std::shared_ptr<TensorType> weight_;
        };

        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_component_cuda_{}_{}.mila", tag, stamp );
        }
    }

    class ComponentSerializationCudaTests : public testing::Test
    {
    protected:
        void SetUp() override
        {
            has_cuda_ = DeviceRegistry::instance().hasDeviceType( DeviceType::Cuda );
        }

        bool has_cuda_ = false;
    };

    // ====================================================================
    // Device parameter staging
    // ====================================================================

    TEST_F( ComponentSerializationCudaTests, SaveParameterToArchive_StagesDeviceParameterAtItsOwnDtype )
    {
        if ( !has_cuda_ )
        {
            GTEST_SKIP() << "No CUDA device present";
        }

        const auto path = makeTempArchivePath( "bf16" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        constexpr dim_t kRows = 2;
        constexpr dim_t kColumns = 3;
        constexpr dim_t kElements = kRows * kColumns;

        auto exec_context = createExecutionContext( Device::Cuda( 0 ) );

        CudaParameterComponent component( "bf16_leaf", Device::Cuda( 0 ), shape_t{ kRows, kColumns } );
        component.exposeSetExecutionContext( exec_context.get() );

        // 1.5f is chosen so the two encodings are distinguishable byte for byte:
        // FP32 1.5 is 0x3FC00000, BF16 1.5 is 0x3FC0.
        fill( component.weight(), 1.5f, exec_context.get() );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            component.save_( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        ASSERT_TRUE( reader.hasFile( "tensors/weight/meta.json" ) );
        ASSERT_TRUE( reader.hasFile( "tensors/weight/data.bin" ) );

        const SerializationMetadata meta = reader.readMetadata( "tensors/weight/meta.json" );

        EXPECT_EQ( meta.getString( "dtype" ), "BF16" );
        EXPECT_EQ( meta.getShape( "shape" ), ( shape_t{ kRows, kColumns } ) );

        // Two bytes per element, not four: the blob carries the parameter's own storage
        // bytes rather than a widened copy.
        constexpr size_t kExpectedBytes = static_cast<size_t>( kElements ) * 2;

        EXPECT_EQ( meta.getInt( "total_bytes" ), static_cast<int64_t>( kExpectedBytes ) );
        EXPECT_EQ( reader.getFileSize( "tensors/weight/data.bin" ), kExpectedBytes );

        // The byte count alone does not discriminate -- staging through an FP32 buffer
        // and writing the BF16 count produced a file of exactly this size, holding the
        // first half of the FP32 data. Reading the contents does: BF16 1.5 gives six
        // identical 0x3FC0 units, where a truncated FP32 buffer would alternate
        // 0x0000 / 0x3FC0.
        std::vector<std::uint16_t> units( kElements, 0 );
        const size_t bytes_read = reader.readBlobInto(
            "tensors/weight/data.bin", units.data(), kExpectedBytes );

        ASSERT_EQ( bytes_read, kExpectedBytes );

        for ( size_t i = 0; i < units.size(); ++i )
        {
            EXPECT_EQ( units[ i ], 0x3FC0u ) << "element " << i;
        }

        std::filesystem::remove( path, ec );
    }

    TEST_F( ComponentSerializationCudaTests, RequireSerializableParameters_PassesForNamedDeviceParameters )
    {
        if ( !has_cuda_ )
        {
            GTEST_SKIP() << "No CUDA device present";
        }

        CudaParameterComponent component( "bf16_leaf", Device::Cuda( 0 ), shape_t{ 2, 3 } );

        ASSERT_GT( component.parameterCount(), 0 );
        EXPECT_NO_THROW( component.requireSerializableParameters() );
    }
}
