#include <gtest/gtest.h>
#include <vector>
#include <string>

import Mila;

namespace Tensors::Tests
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
        Tensor<float, Compute::HostMemoryResource> tensor1;
        Tensor<float, Compute::HostMemoryResource> tensor2;

        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_WithShape ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor1( shape );
        Tensor<float, Compute::HostMemoryResource> tensor2( shape );

        EXPECT_NE( tensor1.getUId(), tensor2.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_DifferentMemoryTypes ) {
        std::vector<size_t> shape = { 2, 3 };

        Tensor<float, Compute::HostMemoryResource> host_tensor( shape );
        Tensor<float, Compute::CudaMemoryResource> cuda_tensor( shape );
        Tensor<float, Compute::CudaPinnedMemoryResource> pinned_tensor( shape );
        Tensor<float, Compute::CudaManagedMemoryResource> managed_tensor( shape );

        EXPECT_NE( host_tensor.getUId(), cuda_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), managed_tensor.getUId() );
        EXPECT_NE( cuda_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( cuda_tensor.getUId(), managed_tensor.getUId() );
        EXPECT_NE( pinned_tensor.getUId(), managed_tensor.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_DifferentDataTypes ) {
        std::vector<size_t> shape = { 2, 3 };

        Tensor<float, Compute::HostMemoryResource> float_tensor( shape );
        Tensor<int, Compute::HostMemoryResource> int_tensor( shape );
        Tensor<uint16_t, Compute::HostMemoryResource> uint16_tensor( shape );
        Tensor<int16_t, Compute::HostMemoryResource> int16_tensor( shape );
        Tensor<uint32_t, Compute::HostMemoryResource> uint32_tensor( shape );

        EXPECT_NE( float_tensor.getUId(), int_tensor.getUId() );
        EXPECT_NE( float_tensor.getUId(), uint16_tensor.getUId() );
        EXPECT_NE( float_tensor.getUId(), int16_tensor.getUId() );
        EXPECT_NE( float_tensor.getUId(), uint32_tensor.getUId() );
        EXPECT_NE( int_tensor.getUId(), uint16_tensor.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_Format ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        std::string uid = tensor.getUId();

        EXPECT_TRUE( uid.find( "tensor_" ) == 0 );
        EXPECT_GT( uid.length(), 7 );

        std::string number_part = uid.substr( 7 );
        EXPECT_FALSE( number_part.empty() );

        for ( char c : number_part ) {
            EXPECT_TRUE( std::isdigit( c ) );
        }
    }

    TEST( TensorIdentityMetadataTest, GetUID_PreservedInMove ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape );
        std::string original_uid = original.getUId();

        Tensor<float, Compute::HostMemoryResource> moved( std::move( original ) );
        EXPECT_EQ( moved.getUId(), original_uid );
    }

    TEST( TensorIdentityMetadataTest, GetUID_PreservedInMoveAssignment ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape );
        std::string original_uid = original.getUId();

        Tensor<float, Compute::HostMemoryResource> moved;
        moved = std::move( original );
        EXPECT_EQ( moved.getUId(), original_uid );
    }

    TEST( TensorIdentityMetadataTest, GetUID_NewInClone ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape );

        auto cloned = original.clone();
        EXPECT_NE( original.getUId(), cloned.getUId() );
    }

    TEST( TensorIdentityMetadataTest, GetUID_NewInMemoryTransfer ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> host_tensor( shape );

        auto cuda_tensor = host_tensor.toDevice<Compute::CudaMemoryResource>();
        EXPECT_NE( host_tensor.getUId(), cuda_tensor.getUId() );

        auto back_to_host = cuda_tensor.toHost<Compute::HostMemoryResource>();
        EXPECT_NE( host_tensor.getUId(), back_to_host.getUId() );
        EXPECT_NE( cuda_tensor.getUId(), back_to_host.getUId() );
    }

    // ====================================================================
    // Tensor Name (getName/setName) Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, GetName_DefaultEmpty ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        EXPECT_TRUE( tensor.getName().empty() );
    }

    TEST( TensorIdentityMetadataTest, SetName_BasicFunctionality ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        tensor.setName( "test_tensor" );
        EXPECT_EQ( tensor.getName(), "test_tensor" );
    }

    TEST( TensorIdentityMetadataTest, SetName_OverwriteExisting ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        tensor.setName( "first_name" );
        EXPECT_EQ( tensor.getName(), "first_name" );

        tensor.setName( "second_name" );
        EXPECT_EQ( tensor.getName(), "second_name" );
    }

    TEST( TensorIdentityMetadataTest, SetName_EmptyStringThrows ) {
        Tensor<float, Compute::HostMemoryResource> tensor;
        EXPECT_THROW( tensor.setName( "" ), std::invalid_argument );
    }

    TEST( TensorIdentityMetadataTest, SetName_WhitespaceOnly ) {
        Tensor<float, Compute::HostMemoryResource> tensor;

        EXPECT_NO_THROW( tensor.setName( " " ) );
        EXPECT_EQ( tensor.getName(), " " );

        EXPECT_NO_THROW( tensor.setName( "  \t  " ) );
        EXPECT_EQ( tensor.getName(), "  \t  " );
    }

    TEST( TensorIdentityMetadataTest, SetName_SpecialCharacters ) {
        Tensor<float, Compute::HostMemoryResource> tensor;

        std::string special_name = "tensor_123!@#$%^&*()_+-=[]{}|;':\",./<>?";
        tensor.setName( special_name );
        EXPECT_EQ( tensor.getName(), special_name );
    }

    TEST( TensorIdentityMetadataTest, SetName_LongName ) {
        Tensor<float, Compute::HostMemoryResource> tensor;

        std::string long_name( 1000, 'a' );
        tensor.setName( long_name );
        EXPECT_EQ( tensor.getName(), long_name );
    }

    TEST( TensorIdentityMetadataTest, SetName_Unicode ) {
        Tensor<float, Compute::HostMemoryResource> tensor;

        std::string unicode_name = "??????_??_????";
        tensor.setName( unicode_name );
        EXPECT_EQ( tensor.getName(), unicode_name );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInMove ) {
        Tensor<float, Compute::HostMemoryResource> original;
        original.setName( "movable_tensor" );

        Tensor<float, Compute::HostMemoryResource> moved( std::move( original ) );
        EXPECT_EQ( moved.getName(), "movable_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInMoveAssignment ) {
        Tensor<float, Compute::HostMemoryResource> original;
        original.setName( "assignable_tensor" );

        Tensor<float, Compute::HostMemoryResource> target;
        target = std::move( original );
        EXPECT_EQ( target.getName(), "assignable_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInClone ) {
        Tensor<float, Compute::HostMemoryResource> original;
        original.setName( "cloneable_tensor" );

        auto cloned = original.clone();
        EXPECT_EQ( cloned.getName(), "cloneable_tensor" );

        cloned.setName( "modified_clone" );
        EXPECT_EQ( original.getName(), "cloneable_tensor" );
        EXPECT_EQ( cloned.getName(), "modified_clone" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInMemoryTransfer ) {
        Tensor<float, Compute::HostMemoryResource> host_tensor;
        host_tensor.setName( "transferable_tensor" );

        auto cuda_tensor = host_tensor.toDevice<Compute::CudaMemoryResource>();
        EXPECT_EQ( cuda_tensor.getName(), "transferable_tensor" );

        auto back_to_host = cuda_tensor.toHost<Compute::HostMemoryResource>();
        EXPECT_EQ( back_to_host.getName(), "transferable_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Name_PreservedInShapeOperations ) {
        std::vector<size_t> shape = { 2, 3, 4 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
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
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.setName( "fillable_tensor" );

        tensor.fill( 1.0f );
        EXPECT_EQ( tensor.getName(), "fillable_tensor" );

        tensor.set( { 0, 0 }, 2.0f );
        EXPECT_EQ( tensor.getName(), "fillable_tensor" );
    }

    // ====================================================================
    // Cross-Memory Resource Identity Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, CrossMemoryAssignment_CreatesNewTensor ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> host_tensor( shape );
        host_tensor.setName( "host_source" );
        std::string host_uid = host_tensor.getUId();

        auto cuda_tensor = host_tensor.toDevice<Compute::CudaMemoryResource>();

        EXPECT_EQ( host_tensor.getUId(), host_uid );
        EXPECT_NE( cuda_tensor.getUId(), host_uid );
        EXPECT_EQ( cuda_tensor.getName(), "host_source" );
    }

    TEST( TensorIdentityMetadataTest, CrossMemoryMoveAssignment_PreservesTargetUID ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> host_tensor( shape );
        host_tensor.setName( "host_source" );

        Tensor<float, Compute::CudaMemoryResource> cuda_tensor( shape );
        cuda_tensor.setName( "cuda_target" );
        std::string cuda_uid = cuda_tensor.getUId();

        cuda_tensor = host_tensor.toDevice<Compute::CudaMemoryResource>();

        EXPECT_NE( cuda_tensor.getUId(), cuda_uid );
        EXPECT_EQ( cuda_tensor.getName(), "host_source" );
    }

    // ====================================================================
    // Identity Consistency Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, Identity_ConsistencyAfterOperations ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> tensor( shape );
        tensor.setName( "consistent_tensor" );
        std::string original_uid = tensor.getUId();

        tensor.fill( 1.0f );
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        tensor.reshape( { 3, 2 } );
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        tensor.flatten();
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );

        tensor[ 0, 0 ] = 2.0f;
        EXPECT_EQ( tensor.getUId(), original_uid );
        EXPECT_EQ( tensor.getName(), "consistent_tensor" );
    }

    TEST( TensorIdentityMetadataTest, Identity_IndependentAfterClone ) {
        std::vector<size_t> shape = { 2, 3 };
        Tensor<float, Compute::HostMemoryResource> original( shape );
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
        // Tensor<float, Compute::HostMemoryResource> original(shape);
        // Tensor<float, Compute::HostMemoryResource> copy(original);  // Should not compile
        // Tensor<float, Compute::HostMemoryResource> assigned;
        // assigned = original;  // Should not compile

        SUCCEED();
    }

    // ====================================================================
    // Edge Cases and Error Conditions
    // ====================================================================

    TEST( TensorIdentityMetadataTest, Name_EmptyTensorOperations ) {
        Tensor<float, Compute::HostMemoryResource> empty_tensor;
        empty_tensor.setName( "empty_tensor" );

        EXPECT_EQ( empty_tensor.getName(), "empty_tensor" );
        EXPECT_FALSE( empty_tensor.getUId().empty() );

        auto cloned_empty = empty_tensor.clone();
        EXPECT_EQ( cloned_empty.getName(), "empty_tensor" );
        EXPECT_NE( cloned_empty.getUId(), empty_tensor.getUId() );
    }

    TEST( TensorIdentityMetadataTest, Name_SingleElementTensor ) {
        std::vector<size_t> shape = { 1 };
        Tensor<float, Compute::HostMemoryResource> single_tensor( shape );
        single_tensor.setName( "single_element" );

        EXPECT_EQ( single_tensor.getName(), "single_element" );

        single_tensor[ 0 ] = 99.0f;
        EXPECT_EQ( single_tensor.getName(), "single_element" );
    }

    TEST( TensorIdentityMetadataTest, UID_MonotonicIncreasing ) {
        std::vector<size_t> tensor_ids;

        for ( int i = 0; i < 10; ++i ) {
            Tensor<float, Compute::HostMemoryResource> tensor;
            std::string uid = tensor.getUId();
            std::string number_part = uid.substr( 7 );
            size_t id = std::stoull( number_part );
            tensor_ids.push_back( id );
        }

        for ( size_t i = 1; i < tensor_ids.size(); ++i ) {
            EXPECT_GT( tensor_ids[ i ], tensor_ids[ i - 1 ] );
        }
    }

    // ====================================================================
    // Thread Safety Tests (Basic)
    // ====================================================================

    TEST( TensorIdentityMetadataTest, UID_ThreadSafety_Sequential ) {
        std::vector<std::string> uids;

        for ( int i = 0; i < 100; ++i ) {
            Tensor<float, Compute::HostMemoryResource> tensor;
            uids.push_back( tensor.getUId() );
        }

        for ( size_t i = 0; i < uids.size(); ++i ) {
            for ( size_t j = i + 1; j < uids.size(); ++j ) {
                EXPECT_NE( uids[ i ], uids[ j ] );
            }
        }
    }

    // ====================================================================
    // Type Alias Identity Tests
    // ====================================================================

    TEST( TensorIdentityMetadataTest, TypeAliases_UniqueIdentities ) {
        std::vector<size_t> shape = { 2, 3 };

        HostTensor<float> host_tensor( shape );
        DeviceTensor<float> device_tensor( shape );
        PinnedTensor<float> pinned_tensor( shape );
        UniversalTensor<float> universal_tensor( shape );

        host_tensor.setName( "host_tensor" );
        device_tensor.setName( "device_tensor" );
        pinned_tensor.setName( "pinned_tensor" );
        universal_tensor.setName( "universal_tensor" );

        EXPECT_NE( host_tensor.getUId(), device_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( host_tensor.getUId(), universal_tensor.getUId() );
        EXPECT_NE( device_tensor.getUId(), pinned_tensor.getUId() );
        EXPECT_NE( device_tensor.getUId(), universal_tensor.getUId() );
        EXPECT_NE( pinned_tensor.getUId(), universal_tensor.getUId() );

        EXPECT_EQ( host_tensor.getName(), "host_tensor" );
        EXPECT_EQ( device_tensor.getName(), "device_tensor" );
        EXPECT_EQ( pinned_tensor.getName(), "pinned_tensor" );
        EXPECT_EQ( universal_tensor.getName(), "universal_tensor" );
    }
}