/**
 * @file GqaWorkspace.ixx
 * @brief The owning GQA transient scratch one attention stack shares, and its factory.
 *
 * GqaState is the non-owning view of it that the operation takes. Family-neutral, so a transformer and a
 * layer-streamed harness allocate the identical set.
 */

module;
#include <cstddef>
#include <memory>
#include <string>

export module Compute.GqaWorkspace;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.GqaState;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The GQA transient the attention layers of one stack share, owned as one unit.
     *
     * Owned together rather than as seven loose tensors so a caller cannot allocate half of it:
     * `GqaState` is a struct of raw pointers, and a null one is a crash at the kernel rather than an
     * error at the call.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    struct GqaWorkspace
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        std::unique_ptr<TensorType> q_permute;
        std::unique_ptr<TensorType> preatt;
        std::unique_ptr<TensorType> att;
        std::unique_ptr<TensorType> v_out;
        std::unique_ptr<TensorType> preatt_decode;
        std::unique_ptr<TensorType> att_decode;
        std::unique_ptr<TensorType> v_out_decode;

        GqaState state() const
        {
            GqaState gqa_state;
            gqa_state.q_permute = q_permute.get();
            gqa_state.preatt = preatt.get();
            gqa_state.att = att.get();
            gqa_state.v_out = v_out.get();
            gqa_state.preatt_decode = preatt_decode.get();
            gqa_state.att_decode = att_decode.get();
            gqa_state.v_out_decode = v_out_decode.get();

            return gqa_state;
        }

        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const auto* t : { q_permute.get(), preatt.get(), att.get(), v_out.get(),
                                    preatt_decode.get(), att_decode.get(), v_out_decode.get() } )
            {
                if ( t )
                    total += occupiedTensorBytes( *t );
            }

            return total;
        }
    };

    /**
     * @brief Allocate the shared GQA transient.
     *
     * @param head_dim    The widest head any layer of the stack uses; narrower layers view a prefix.
     * @param score_width The caller's flash decision made concrete: the flash path reads no score buffer,
     *                    while the cuBLASLt path needs the full context and would overflow a narrow one.
     *                    Taken explicitly so a caller cannot allocate for one path and then run the other.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    GqaWorkspace<TDeviceType, TPrecision> makeGqaWorkspace(
        DeviceId device, dim_t B, dim_t num_heads, dim_t head_dim, dim_t T_ctx,
        dim_t prefill_chunk, dim_t score_width, const std::string& name_prefix )
    {
        using TensorType = typename GqaWorkspace<TDeviceType, TPrecision>::TensorType;

        GqaWorkspace<TDeviceType, TPrecision> workspace;

        workspace.q_permute = std::make_unique<TensorType>(
            device, shape_t{ B, num_heads, prefill_chunk, head_dim }, name_prefix + "q_perm" );
        workspace.preatt = std::make_unique<TensorType>(
            device, shape_t{ B, num_heads, prefill_chunk, score_width }, name_prefix + "preatt" );
        workspace.att = std::make_unique<TensorType>(
            device, shape_t{ B, num_heads, prefill_chunk, score_width }, name_prefix + "att" );
        workspace.v_out = std::make_unique<TensorType>(
            device, shape_t{ B, num_heads, prefill_chunk, head_dim }, name_prefix + "v_out" );

        workspace.preatt_decode = std::make_unique<TensorType>(
            device, shape_t{ B, num_heads, 1, T_ctx }, name_prefix + "preatt_dec" );
        workspace.att_decode = std::make_unique<TensorType>(
            device, shape_t{ B, num_heads, 1, T_ctx }, name_prefix + "att_dec" );
        workspace.v_out_decode = std::make_unique<TensorType>(
            device, shape_t{ B, num_heads, 1, head_dim }, name_prefix + "v_out_dec" );

        return workspace;
    }

    /**
     * @brief Device bytes makeGqaWorkspace would allocate with the same arguments, without allocating.
     *
     * @param granularity The allocation granularity of the device the workspace would live on.
     */
    export template<TensorDataType TPrecision>
    std::size_t gqaWorkspaceDeviceBytes( std::size_t granularity, dim_t B, dim_t num_heads, dim_t head_dim,
        dim_t T_ctx, dim_t prefill_chunk, dim_t score_width )
    {
        const auto occupied = [&]( dim_t elements )
        {
            return occupiedDeviceBytes(
                static_cast<std::size_t>( elements ) * TensorDataTypeTraits<TPrecision>::size_in_bytes, granularity );
        };

        return 2 * occupied( B * num_heads * prefill_chunk * head_dim )    // q_permute, v_out
            + 2 * occupied( B * num_heads * prefill_chunk * score_width )  // preatt, att
            + 2 * occupied( B * num_heads * T_ctx )                        // preatt_decode, att_decode
            + occupied( B * num_heads * head_dim );                        // v_out_decode
    }
}
