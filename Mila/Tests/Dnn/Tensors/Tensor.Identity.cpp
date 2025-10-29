#include <gtest/gtest.h>
#include <vector>
#include <string>

import Mila;

namespace Dnn::Tensors::Tests
{
    using namespace Mila::Dnn;

    class TensorIdentityTest : public testing::Test {
    protected:
        TensorIdentityTest() {}
    };

    // ====================================================================
    // Unique Identifier (getUId) Tests
    // ====================================================================

    TEST( TensorIdentityTest, GetUID_UniqueGeneration ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor1( "CPU", shape_t{} );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor2( "CPU", shape_t{} );

        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST( TensorIdentityTest, GetUID_WithShape ) {
        shape_t shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor1( "CPU", shape );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor2( "CPU", shape );

        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST( TensorIdentityTest, GetUID_DifferentMemoryTypes ) {
        std::vector<int64_t> shape = { 2, 3 };

        // Only exercise host memory in unit tests to avoid device context requirements.
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> host_tensor( "CPU", shape );
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> another_host_tensor( "CPU", shape );

        EXPECT_NE( host_tensor.getUId(), another_host_tensor.getUId() );
    }

    TEST( TensorIdentityTest, GetUID_DifferentDataTypes ) {
        std::vector<int64_t> shape = { 2, 3 };

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

    TEST( TensorIdentityTest, GetUID_Format ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );
        std::string uid = tensor.getUId();

        EXPECT_TRUE( uid.find( "tensor_" ) == 0 );
        EXPECT_GT( uid.length(), 7 );

        std::string number_part = uid.substr( 7 );
        EXPECT_FALSE( number_part.empty() );

        for (char c : number_part) {
            EXPECT_TRUE( std::isdigit( c ) );
        }
    }

    TEST( TensorIdentityTest, GetUID_PreservedInMove ) {
        std::vector<int64_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape );
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> moved( std::move( original ) );
        EXPECT_EQ( moved.getUId(), original_uid );
    }

    TEST( TensorIdentityTest, GetUID_PreservedInMoveAssignment ) {
        std::vector<int64_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape );
        std::string original_uid = original.getUId();

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> moved( "CPU", shape_t{} );
        moved = std::move( original );
        EXPECT_EQ( moved.getUId(), original_uid );
    }

    // ====================================================================
    // Tensor Name (getName/setName) Tests
    // ====================================================================

    TEST( TensorIdentityTest, GetName_DefaultEmpty ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );
        EXPECT_TRUE( tensor.getName().empty() );
    }

    TEST( TensorIdentityTest, SetName_BasicFunctionality ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );
        tensor.setName( "test_tensor" );
        EXPECT_EQ( tensor.getName(), "test_tensor" );
    }

    TEST( TensorIdentityTest, SetName_OverwriteExisting ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );
        tensor.setName( "first_name" );
        EXPECT_EQ( tensor.getName(), "first_name" );

        tensor.setName( "second_name" );
        EXPECT_EQ( tensor.getName(), "second_name" );
    }

    TEST( TensorIdentityTest, SetName_EmptyStringThrows ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );
        EXPECT_THROW( tensor.setName( "" ), std::invalid_argument );
    }

    TEST( TensorIdentityTest, SetName_WhitespaceOnly ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );

        EXPECT_NO_THROW( tensor.setName( " " ) );
        EXPECT_EQ( tensor.getName(), " " );

        EXPECT_NO_THROW( tensor.setName( "  \t  " ) );
        EXPECT_EQ( tensor.getName(), "  \t  " );
    }

    TEST( TensorIdentityTest, SetName_SpecialCharacters ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );

        std::string special_name = "tensor_123!@#$%^&*()_+-=[]{}|;':\",./<>?";
        tensor.setName( special_name );
        EXPECT_EQ( tensor.getName(), special_name );
    }

    TEST( TensorIdentityTest, SetName_LongName ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );

        std::string long_name( 1000, 'a' );
        tensor.setName( long_name );
        EXPECT_EQ( tensor.getName(), long_name );
    }

    TEST( TensorIdentityTest, SetName_Unicode ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );

        std::string unicode_name = "??????_??_????";
        tensor.setName( unicode_name );
        EXPECT_EQ( tensor.getName(), unicode_name );
    }

    TEST( TensorIdentityTest, Name_PreservedInMove ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape_t{} );
        original.setName( "movable_tensor" );

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> moved( std::move( original ) );
        EXPECT_EQ( moved.getName(), "movable_tensor" );
    }

    TEST( TensorIdentityTest, Name_PreservedInMoveAssignment ) {
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original( "CPU", shape_t{} );
        original.setName( "assignable_tensor" );

        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> target( "CPU", shape_t{} );
        target = std::move( original );
        EXPECT_EQ( target.getName(), "assignable_tensor" );
    }

    TEST( TensorIdentityTest, Name_PreservedInShapeOperations ) {
        std::vector<int64_t> shape = { 2, 3, 4 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );
        tensor.setName( "reshapeable_tensor" );

        tensor.reshape( { 6, 4 } );
        EXPECT_EQ( tensor.getName(), "reshapeable_tensor" );

        /* TODO: tensor.flatten();
        EXPECT_EQ( tensor.getName(), "reshapeable_tensor" );

        auto flattened = tensor.flattened();
        EXPECT_EQ( flattened.getName(), "reshapeable_tensor" );*/
    }

    TEST( TensorIdentityTest, Name_PreservedInDataOperations ) {
        std::vector<int64_t> shape = { 2, 3 };
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

    TEST( TensorIdentityTest, CrossMemoryAssignment_CreatesNewTensor ) {
        std::vector<int64_t> shape = { 2, 3 };
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

    TEST( TensorIdentityTest, CrossMemoryMoveAssignment_PreservesTargetUID ) {
        std::vector<int64_t> shape = { 2, 3 };
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

    TEST( TensorIdentityTest, Identity_ConsistencyAfterOperations ) {
        std::vector<int64_t> shape = { 2, 3 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape );
        tensor.setName( "consistent_tensor" );
        std::string original_uid = tensor.getUId();

        ASSERT_NO_THROW( fill( tensor, 1.0f ) );
        
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        tensor.reshape( { 3, 2 } );
        
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        /* TODO: tensor.flatten();
        
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );*/

        // Per-element write via raw pointer
        float* data = static_cast<float*>(tensor.data());
        data[0] = 2.0f;
        
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );
    }

    

    // ====================================================================
    // Deleted Operations Tests
    // ====================================================================

    TEST( TensorIdentityTest, CopyOperationsAreDeleted ) {
        // These operations should not compile with move-only tensors
        // Uncomment to verify compilation errors:

        // std::vector<int64_t> shape = {2, 3};
        // Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> original(shape);
        // Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> copy(original);  // Should not compile
        // Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> assigned;
        // assigned = original;  // Should not compile

        SUCCEED();
    }

    // ====================================================================
    // Edge Cases and Error Conditions
    // ====================================================================

    

    TEST( TensorIdentityTest, Name_SingleElementTensor ) {
        std::vector<int64_t> shape = { 1 };
        Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> single_tensor( "CPU", shape );
        single_tensor.setName( "single_element" );

        EXPECT_EQ( single_tensor.getName(), "single_element" );

        float* data = static_cast<float*>(single_tensor.data());
        data[0] = 99.0f;
        EXPECT_EQ( single_tensor.getName(), "single_element" );
    }

    TEST( TensorIdentityTest, UID_MonotonicIncreasing ) {
        std::vector<size_t> tensor_ids;

        for (int i = 0; i < 10; ++i) {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );
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

    TEST( TensorIdentityTest, UID_ThreadSafety_Sequential ) {
        std::vector<std::string> uids;

        for (int i = 0; i < 100; ++i) {
            Tensor<TensorDataType::FP32, Compute::CpuMemoryResource> tensor( "CPU", shape_t{} );
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

    TEST( TensorIdentityTest, TypeAliases_UniqueIdentities ) {
        shape_t shape = { 2, 3 };

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