#include <gtest/gtest.h>
#include <vector>
#include <string>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorIdentityMetadataTest : public testing::Test {
    protected:
        TensorIdentityMetadataTest() {}
    };

    // ====================================================================
    // Unique Identifier (getUId) Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, GetUID_UniqueGeneration ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor1( "CPU", std::vector<size_t>{} );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor2( "CPU", std::vector<size_t>{} );

        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_WithShape ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor1( "CPU", shape );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor2( "CPU", shape );

        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_DifferentMemoryTypes ) {
        std::vector<size_t> shape = { 2, 3 };

        // Only exercise host memory in unit tests to avoid device context requirements.
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> another_host_tensor( "CPU", shape );

        EXPECT_NE( host_tensor.getUId(), another_host_tensor.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_DifferentDataTypes ) {
        std::vector<size_t> shape = { 2, 3 };

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> float_tensor( "CPU", shape );
        Tensor<TensorDataType::INT32, Compute::CpuMemoryResource> int_tensor( "CPU", shape );
        Tensor<TensorDataType::UINT16, Compute::CpuMemoryResource> uint16_tensor( "CPU", shape );
        Tensor<TensorDataType::INT16, Compute::CpuMemoryResource> int16_tensor( "CPU", shape );
        Tensor<TensorDataType::UINT32, Compute::CpuMemoryResource> uint32_tensor( "CPU", shape );

        EXPECT_NE( float_tensor.getUId(), int_tensor.getUId() );
        EXPECT_NE( float_tensor.getUId(), uint16_tensor.getUId() );
        EXPECT_NE( float_tensor.getUId(), int16_tensor.getUId() );
        EXPECT_NE( float_tensor.getUId(), uint32_tensor.getUId() );
        EXPECT_NE( int_tensor.getUId(), uint16_tensor.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_Format ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );
        std::string uid = tensor.getUId();

        EXPECT_TRUE( uid.find( "tensor_" ) == 0 );
        EXPECT_GT( uid.length(), 7 );

        std::string number_part = uid.substr( 7 );
        EXPECT_FALSE( number_part.empty() );

        for (char c : number_part) {
            EXPECT_TRUE( std::isdigit( c ) );
        }
    }

    TEST( TensorIdentityMetadataTest, GetUID_PreservedInMove ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape );
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> moved( std::move( original ) );
        EXPECT_EQ( moved.getUId(), original_uid );
    }

    TEST( TensorIdentityMetadataTest, GetUID_PreservedInMoveAssignment ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape );
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> moved( "CPU", std::vector<size_t>{} );
        moved = std::move( original );
        EXPECT_EQ( moved.getUId(), original_uid );
    }

    TEST( TensorIdentityMetadataTest, GetUID_NewInClone ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape );

        auto cloned = original.clone();
        EXPECT_NE( original.getUId(), cloned.getUId() );
    }

    // ====================================================================
    // Tensor Name (getName/setName) Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, GetName_DefaultEmpty ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );
        EXPECT_TRUE( tensor.getName().empty() );
    }

    TEST( TensorIdentityMetadataTest, SetName_BasicFunctionality ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );
        tensor.setName( "test_tensor" );
        EXPECT_EQ( tensor.getName(), "test_tensor" );
    }

    TEST( TensorIdentityMetadataTest, SetName_OverwriteExisting ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );
        tensor.setName( "first_name" );
        EXPECT_EQ( tensor.getName(), "first_name" );

        tensor.setName( "second_name" );
        EXPECT_EQ( tensor.getName(), "second_name" );
    }

    TEST( TensorIdentityMetadataTest, SetName_EmptyStringThrows ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );
        EXPECT_THROW( tensor.setName( "" ), std::invalid_argument );
    }

    TEST( TensorIdentityMetadataTest, SetName_WhitespaceOnly ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );

        EXPECT_NO_THROW( tensor.setName( " " ) );
        EXPECT_EQ( tensor.getName(), " " );

        EXPECT_NO_THROW( tensor.setName( "  \t  " ) );
        EXPECT_EQ( tensor.getName(), "  \t  " );
    }

    TEST( TensorIdentityMetadataTest, SetName_SpecialCharacters ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );

        std::string special_name = "tensor_123!@#$%^&*()_+-=[]{}|;':\",./<>?";
        tensor.setName( special_name );
        EXPECT_EQ( tensor.getName(), special_name );
    }

    TEST( TensorIdentityMetadataTest, SetName_LongName ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );

        std::string long_name( 1000, 'a' );
        tensor.setName( long_name );
        EXPECT_EQ( tensor.getName(), long_name );
    }

    TEST( TensorIdentityMetadataTest, SetName_Unicode ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );

        std::string unicode_name = "??????_??_????";
        tensor.setName( unicode_name );
        EXPECT_EQ( tensor.getName(), unicode_name );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInMove ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", std::vector<size_t>{} );
        original.setName( "movable_tensor" );

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> moved( std::move( original ) );
        EXPECT_EQ( moved.getName(), "movable_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInMoveAssignment ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", std::vector<size_t>{} );
        original.setName( "assignable_tensor" );

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> target( "CPU", std::vector<size_t>{} );
        target = std::move( original );
        EXPECT_EQ( target.getName(), "assignable_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInClone ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", std::vector<size_t>{} );
        original.setName( "cloneable_tensor" );

        auto cloned = original.clone();
        EXPECT_EQ( cloned.getName(), "cloneable_tensor" );

        cloned.setName( "modified_clone" );
        EXPECT_EQ( original.getName(), "cloneable_tensor" );
        EXPECT_EQ( cloned.getName(), "modified_clone" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInShapeOperations ) {
        std::vector<size_t> shape = { 2, 3, 4 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );
        tensor.setName( "reshapeable_tensor" );

        tensor.reshape( { 6, 4 } );
        EXPECT_EQ( tensor.getName(), "reshapeable_tensor" );

        tensor.flatten();
        EXPECT_EQ( tensor.getName(), "reshapeable_tensor" );

        auto flattened = tensor.flattened();
        EXPECT_EQ( flattened.getName(), "reshapeable_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInDataOperations ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );
        tensor.setName( "fillable_tensor" );

        ASSERT_NO_THROW( fill( tensor, 1.0f ) );
        EXPECT_EQ( tensor.getName(), "fillable_tensor" );

        // Per-element set via rawData for host tensors
        float* data = static_cast<float*>(tensor.data());
        data[0] = 2.0f;
        EXPECT_EQ( tensor.getName(), "fillable_tensor" );
    }

    // ====================================================================
    // Cross-Memory Resource Identity Tests (simplified for CPU-only tests)
    // ====================================================================

    TEST( TensorIdentityMetadataTest, CrossMemoryAssignment_CreatesNewTensor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
        host_tensor.setName( "host_source" );
        std::string host_uid = host_tensor.getUId();

        // Simulate a new tensor that would represent a cross-memory copy
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> new_tensor( "CPU", shape );
        new_tensor.setName( "host_source" );

        EXPECT_EQ( host_tensor.getUId(), host_uid );
        EXPECT_NE( new_tensor.getUId(), host_uid );
        EXPECT_EQ( new_tensor.getName(), "host_source" );
    }

    TEST( TensorIdentityMetadataTest, CrossMemoryMoveAssignment_PreservesTargetUID ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
        host_tensor.setName( "host_source" );

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> target( "CPU", shape );
        target.setName( "cpu_target" );
        std::string target_uid = target.getUId();

        // Simulate assignment by move from a new tensor
        target = std::move( host_tensor );
        EXPECT_NE( target.getUId(), target_uid );
        EXPECT_EQ( target.getName(), "host_source" );
    }

    // ====================================================================
    // Identity Consistency Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, Identity_ConsistencyAfterOperations ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );
        tensor.setName( "consistent_tensor" );
        std::string original_uid = tensor.getUId();

        ASSERT_NO_THROW( fill( tensor, 1.0f ) );
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        tensor.reshape( { 3, 2 } );
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        tensor.flatten();
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        // Per-element write via raw pointer
        float* data = static_cast<float*>(tensor.data());
        data[0] = 2.0f;
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Identity_IndependentAfterClone ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape );
        original.setName( "original_tensor" );

        auto cloned = original.clone();

        original.setName( "modified_original" );
        EXPECT_EQ( original.getName(), "modified_original" );
        EXPECT_EQ( cloned.getName(), "original_tensor" );

        cloned.setName( "modified_clone" );
        EXPECT_EQ( original.getName(), "modified_original" );
        EXPECT_EQ( cloned.getName(), "modified_clone" );
    }

    // ====================================================================
    // Deleted Operations Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, CopyOperationsAreDeleted ) {
        // These operations should not compile with move-only tensors
        // Uncomment to verify compilation errors:

        // std::vector<size_t> shape = {2, 3};
        // Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original(shape);
        // Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> copy(original);  // Should not compile
        // Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> assigned;
        // assigned = original;  // Should not compile

        SUCCEED();
    }

    // ====================================================================
    // Edge Cases and Error Conditions
    // ====================================================================

    TEST( TensorIdentityMetadataTest, Name_EmptyTensorOperations ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> empty_tensor( "CPU", std::vector<size_t>{} );
        empty_tensor.setName( "empty_tensor" );

        EXPECT_EQ( empty_tensor.getName(), "empty_tensor" );
        EXPECT_FALSE( empty_tensor.getUId().empty() );

        auto cloned_empty = empty_tensor.clone();
        EXPECT_EQ( cloned_empty.getName(), "empty_tensor" );
        EXPECT_NE( cloned_empty.getUId(), empty_tensor.getUId() );
    }

    TEST( TensorIdentityMetadataTest, Name_SingleElementTensor ) {
        std::vector<size_t> shape = { 1 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> single_tensor( "CPU", shape );
        single_tensor.setName( "single_element" );

        EXPECT_EQ( single_tensor.getName(), "single_element" );

        float* data = static_cast<float*>(single_tensor.data());
        data[0] = 99.0f;
        EXPECT_EQ( single_tensor.getName(), "single_element" );
    }

    TEST( TensorIdentityMetadataTest, UID_MonotonicIncreasing ) {
        std::vector<size_t> tensor_ids;

        for (int i = 0; i < 10; ++i) {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );
            std::string uid = tensor.getUId();
            std::string number_part = uid.substr( 7 );
            size_t id = std::stoull( number_part );
            tensor_ids.push_back( id );
        }

        for (size_t i = 1; i < tensor_ids.size(); ++i) {
            EXPECT_GT( tensor_ids[i], tensor_ids[i - 1] );
        }
    }

    // ====================================================================
    // Thread Safety Tests (Basic)
    // ====================================================================

    TEST( TensorIdentityMetadataTest, UID_ThreadSafety_Sequential ) {
        std::vector<std::string> uids;

        for (int i = 0; i < 100; ++i) {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", std::vector<size_t>{} );
            uids.push_back( tensor.getUId() );
        }

        for (size_t i = 0; i < uids.size(); ++i) {
            for (size_t j = i + 1; j < uids.size(); ++j) {
                EXPECT_NE( uids[i], uids[j] );
            }
        }
    }

    // ====================================================================
    // Type Alias Identity Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, TypeAliases_UniqueIdentities ) {
        std::vector<size_t> shape = { 2, 3 };

        HostTensor<TensorDataType::FP32> host_tensor( "CPU", shape );
        HostTensor<TensorDataType::FP32> another_host_tensor( "CPU", shape );
        PinnedTensor<TensorDataType::FP32> pinned_tensor( "CUDA:0", shape );
        UniversalTensor<TensorDataType::FP32> universal_tensor( "CUDA:0", shape );

        host_tensor.setName( "host_tensor" );
        another_host_tensor.setName( "device_tensor" );
        pinned_tensor.setName( "pinned_tensor" );
        universal_tensor.setName( "universal_tensor" );

        EXPECT_NE( host_tensor.getUId(), another_host_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), universal_tensor.getUId() );
        EXPECT_NE( another_host_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( another_host_tensor.getUId(), universal_tensor.getUId() );
        EXPECT_NE( pinned_tensor.getUId(), universal_tensor.getUId() );

        EXPECT_EQ( host_tensor.getName(), "host_tensor" );
        EXPECT_EQ( another_host_tensor.getName(), "device_tensor" );
        EXPECT_EQ( pinned_tensor.getName(), "pinned_tensor" );
        EXPECT_EQ( universal_tensor.getName(), "universal_tensor" );
    }
}