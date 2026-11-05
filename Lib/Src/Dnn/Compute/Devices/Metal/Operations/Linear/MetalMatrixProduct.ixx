/**
 * @file MetalMatrixProduct.ixx
 * @brief output = input * weight^T on Metal, on the path Phase 2 measured for each number of rows.
 */

export module Compute.MetalMatrixProduct;

import std;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalMatrixView;
import Compute.MetalTensorDataTypeMap;
import Dnn.TensorDataType;

namespace Mila::Dnn::Compute::Metal
{
    /// A product's three matrices. `input` is rows by inputs; `weight` is outputs by inputs; `output` holds outputs at each `output_stride`.
    export struct MatrixProduct
    {
        const void* input{ nullptr };
        const void* weight{ nullptr };
        const void* output{ nullptr };
        int rows{ 0 };
        int inputs{ 0 };
        int outputs{ 0 };
        int output_stride{ 0 };
    };

    // Not anonymous: the templates below instantiate in the modules that call them, which must reach these by name.
    namespace MatrixProductDetail
    {
        constexpr int kLargeTileRows = 128;
        constexpr int kLargeTileSimdgroups = 4;
        constexpr int kSmallTileRows = 8;
        constexpr int kSmallTileSimdgroups = 2;
        constexpr int kTileColumns = 16;
        constexpr int kLaneOutputs = 4;

        template<TensorDataType TPrecision, int TTileRows, int TSimdgroups>
        int encodeTiles( MetalExecutionContext& context, const MatrixProduct& product, int first_row, int columns )
        {
            const int tiles_down = ( product.rows - first_row ) / TTileRows;

            if ( tiles_down == 0 )
            {
                return first_row;
            }

            const std::size_t element_bytes = metalElementBytes<TPrecision>( 1 );
            const auto* const first_input = static_cast<const std::byte*>( product.input ) + element_bytes * first_row * product.inputs;
            const auto* const first_output = static_cast<const std::byte*>( product.output ) + element_bytes * first_row * product.output_stride;
            const int rows_taken = tiles_down * TTileRows;

            // The weights multiply transposed, so their rows are the outputs here.
            const std::array<MetalMatrixView, 3> matrices{
                MetalMatrixView{ first_input, TPrecision, rows_taken, product.inputs, product.inputs },
                MetalMatrixView{ product.weight, TPrecision, product.outputs, product.inputs, product.inputs },
                MetalMatrixView{ first_output, TPrecision, rows_taken, product.outputs, product.output_stride } };

            const std::string kernel = std::format(
                "linear_tiles_{}_{}_{}_{}", MetalTensorDataTypeMap<TPrecision>::kernel_type, TTileRows, kTileColumns, TSimdgroups );

            context.dispatchMatrices(
                context.device().kernels().pipeline( kernel ), matrices,
                static_cast<std::size_t>( columns / kTileColumns ), static_cast<std::size_t>( tiles_down ), TSimdgroups );

            return first_row + rows_taken;
        }

        /// The own kernel over [first_row, last_row) by [first_output, every output).
        template<TensorDataType TPrecision>
        void encodeOwn( MetalExecutionContext& context, const MatrixProduct& product, int first_row, int last_row, int first_output )
        {
            const int rows = last_row - first_row;
            const int outputs = product.outputs - first_output;

            if ( rows <= 0 || outputs <= 0 )
            {
                return;
            }

            const std::array<const void*, 3> buffers{ product.input, product.weight, product.output };
            const std::string_view type = MetalTensorDataTypeMap<TPrecision>::kernel_type;
            const bool vectors = product.inputs % 4 == 0;

            const std::array<std::uint32_t, 5> rectangle{
                static_cast<std::uint32_t>( product.inputs ), static_cast<std::uint32_t>( product.output_stride ),
                static_cast<std::uint32_t>( first_row ), static_cast<std::uint32_t>( first_output ),
                static_cast<std::uint32_t>( vectors ? product.outputs : outputs ) };

            if ( vectors )
            {
                context.dispatchThreadgroups(
                    context.device().kernels().pipeline( std::format( "linear_lanes_{}", type ) ),
                    buffers, std::as_bytes( std::span( rectangle ) ),
                    static_cast<std::size_t>( ( outputs + kLaneOutputs - 1 ) / kLaneOutputs ), static_cast<std::size_t>( rows ) );

                return;
            }

            context.dispatch(
                context.device().kernels().pipeline( std::format( "linear_elements_{}", type ) ),
                buffers, std::as_bytes( std::span( rectangle ) ),
                static_cast<std::size_t>( rows ) * static_cast<std::size_t>( outputs ) );
        }
    }

    /**
     * @brief Encode one product.
     *
     * Whole tiles of 128 rows and of 8 rows go through TensorOps, 16 outputs to a tile. Everything else, which includes a decode's one
     * row and the rows and outputs a tile leaves over, goes through the own kernel. See `MetalBackend.md` section 4.2.
     */
    export template<TensorDataType TPrecision>
    void encodeMatrixProduct( MetalExecutionContext& context, const MatrixProduct& product )
    {
        using namespace MatrixProductDetail;

        const int columns_in_tiles = product.rows < kSmallTileRows ? 0 : product.outputs - product.outputs % kTileColumns;

        int next_row = 0;

        if ( columns_in_tiles > 0 )
        {
            next_row = encodeTiles<TPrecision, kLargeTileRows, kLargeTileSimdgroups>( context, product, next_row, columns_in_tiles );
            next_row = encodeTiles<TPrecision, kSmallTileRows, kSmallTileSimdgroups>( context, product, next_row, columns_in_tiles );
        }

        // The rows no tile took, across every output, then the outputs no tile took on the rows tiles did.
        encodeOwn<TPrecision>( context, product, next_row, product.rows, 0 );
        encodeOwn<TPrecision>( context, product, 0, next_row, columns_in_tiles );
    }
}
