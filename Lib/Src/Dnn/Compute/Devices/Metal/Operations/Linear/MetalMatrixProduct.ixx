/**
 * @file MetalMatrixProduct.ixx
 * @brief output = input * weight^T on Metal, on the path Phase 2 measured for each number of rows.
 */

export module Compute.MetalMatrixProduct;

import std;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
export import Compute.MetalMatrixProductPipelines;
import Compute.MetalMatrixView;
import Compute.MetalTensorDataTypeMap;
import Dnn.TensorDataType;

namespace Mila::Dnn::Compute::Metal
{
    /// A product that never pads, whatever its size.
    export inline constexpr std::int64_t kNeverPad = std::numeric_limits<std::int64_t>::max();

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

        /// Run the 2 to 7 rows that no tile takes through a tile of 8, by way of the context's scratch buffer, once rows * inputs * outputs
        /// reach this. Below it they go through the own kernel. Where padding pays depends on the shape (`MetalBackend.md` section 4.2).
        std::int64_t fewest_multiplies_to_pad{ kNeverPad };
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
        constexpr int kFewestPaddedRows = 2;

        [[nodiscard]] constexpr int outputsInTiles( int outputs ) noexcept
        {
            return outputs - outputs % kTileColumns;
        }

        [[nodiscard]] constexpr bool padsRows( int rows, int inputs, int outputs, std::int64_t fewest_multiplies ) noexcept
        {
            return rows >= kFewestPaddedRows && outputsInTiles( outputs ) > 0
                && static_cast<std::int64_t>( rows ) * inputs * outputs >= fewest_multiplies;
        }

        template<TensorDataType TPrecision, int TTileRows, int TSimdgroups>
        int encodeTiles( MetalExecutionContext& context, MetalPipeline pipeline, const MatrixProduct& product, int first_row, int columns )
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

            context.dispatchMatrices(
                pipeline, matrices,
                static_cast<std::size_t>( columns / kTileColumns ), static_cast<std::size_t>( tiles_down ), TSimdgroups );

            return first_row + rows_taken;
        }

        /**
         * The rows from `first_row` on, 2 to 7 of them, through one tile of 8 in scratch. A tile reads and writes 8 rows, which the
         * product's own memory need not have. The rows that pad it are zero, so that what they hold depends on nothing that ran earlier.
         */
        template<TensorDataType TPrecision>
        void encodePaddedRows( MetalExecutionContext& context, const MatrixProductPipelines& pipelines, const MatrixProduct& product,
            int first_row, int columns )
        {
            const std::size_t element_bytes = metalElementBytes<TPrecision>( 1 );
            const std::size_t rows = static_cast<std::size_t>( product.rows - first_row );
            const std::size_t input_row_bytes = element_bytes * product.inputs;
            const std::size_t output_row_bytes = element_bytes * columns;
            const std::size_t output_stride_bytes = element_bytes * product.output_stride;
            const std::size_t tile_input_bytes = input_row_bytes * kSmallTileRows;

            auto* const scratch_input = static_cast<std::byte*>(
                context.getDeviceScratchBuffer( tile_input_bytes + output_row_bytes * kSmallTileRows ) );
            auto* const scratch_output = scratch_input + tile_input_bytes;

            context.copyBytes( static_cast<const std::byte*>( product.input ) + input_row_bytes * first_row, scratch_input, input_row_bytes * rows );
            context.zeroBytes( scratch_input + input_row_bytes * rows, input_row_bytes * ( kSmallTileRows - rows ) );

            const MatrixProduct tile{ scratch_input, product.weight, scratch_output, kSmallTileRows, product.inputs, columns, columns };

            encodeTiles<TPrecision, kSmallTileRows, kSmallTileSimdgroups>( context, pipelines.small_tiles, tile, 0, columns );

            // A product names its output by a const address, as every kernel binding does.
            auto* const output = static_cast<std::byte*>( const_cast<void*>( product.output ) ) + output_stride_bytes * first_row;

            for ( std::size_t row = 0; row < rows; ++row )
            {
                context.copyBytes( scratch_output + output_row_bytes * row, output + output_stride_bytes * row, output_row_bytes );
            }
        }

        /// The own kernel over [first_row, last_row) by [first_output, every output).
        template<TensorDataType TPrecision>
        void encodeOwn( MetalExecutionContext& context, const MatrixProductPipelines& pipelines, const MatrixProduct& product,
            int first_row, int last_row, int first_output )
        {
            const int rows = last_row - first_row;
            const int outputs = product.outputs - first_output;

            if ( rows <= 0 || outputs <= 0 )
            {
                return;
            }

            const std::array<const void*, 3> buffers{ product.input, product.weight, product.output };
            const bool vectors = product.inputs % 4 == 0;

            const std::array<std::uint32_t, 5> rectangle{
                static_cast<std::uint32_t>( product.inputs ), static_cast<std::uint32_t>( product.output_stride ),
                static_cast<std::uint32_t>( first_row ), static_cast<std::uint32_t>( first_output ),
                static_cast<std::uint32_t>( vectors ? product.outputs : outputs ) };

            if ( vectors )
            {
                context.dispatchThreadgroups(
                    pipelines.lanes,
                    buffers, std::as_bytes( std::span( rectangle ) ),
                    static_cast<std::size_t>( ( outputs + kLaneOutputs - 1 ) / kLaneOutputs ), static_cast<std::size_t>( rows ) );

                return;
            }

            context.dispatch(
                pipelines.elements,
                buffers, std::as_bytes( std::span( rectangle ) ),
                static_cast<std::size_t>( rows ) * static_cast<std::size_t>( outputs ) );
        }
    }

    /// The most scratch a product of at most `rows` rows asks for: a tile of 8 rows, in and out, or none when it can never pad.
    export template<TensorDataType TPrecision>
    [[nodiscard]] std::size_t matrixProductScratchBytes( int rows, int inputs, int outputs, std::int64_t fewest_multiplies_to_pad )
    {
        using namespace MatrixProductDetail;

        if ( !padsRows( std::min( rows, kSmallTileRows - 1 ), inputs, outputs, fewest_multiplies_to_pad ) )
        {
            return 0;
        }

        return metalElementBytes<TPrecision>( 1 ) * kSmallTileRows * ( static_cast<std::size_t>( inputs ) + outputsInTiles( outputs ) );
    }

    /**
     * @brief The kernels a product in `TPrecision` may dispatch, for the operation that encodes products to resolve when it is constructed.
     *
     * @throws std::runtime_error When Mila has no such kernel, or Metal cannot prepare one.
     */
    export template<TensorDataType TPrecision>
    [[nodiscard]] MatrixProductPipelines matrixProductPipelines( MetalKernelLibrary& kernels )
    {
        using namespace MatrixProductDetail;

        const std::string_view type = MetalTensorDataTypeMap<TPrecision>::kernel_type;

        return MatrixProductPipelines{
            kernels.pipeline( std::format( "linear_tiles_{}_{}_{}_{}", type, kLargeTileRows, kTileColumns, kLargeTileSimdgroups ) ),
            kernels.pipeline( std::format( "linear_tiles_{}_{}_{}_{}", type, kSmallTileRows, kTileColumns, kSmallTileSimdgroups ) ),
            kernels.pipeline( std::format( "linear_lanes_{}", type ) ),
            kernels.pipeline( std::format( "linear_elements_{}", type ) ) };
    }

    /**
     * @brief Encode one product through `pipelines`, which are `matrixProductPipelines<TPrecision>`'s.
     *
     * Whole tiles of 128 rows and of 8 rows go through TensorOps, 16 outputs to a tile, and so do 2 to 7 rows left over when the product
     * is large enough to pad them. Everything else, which includes a decode's one row and the outputs a tile leaves over, goes through the own kernel.
     * See `MetalBackend.md` section 4.2.
     */
    export template<TensorDataType TPrecision>
    void encodeMatrixProduct( MetalExecutionContext& context, const MatrixProductPipelines& pipelines, const MatrixProduct& product )
    {
        using namespace MatrixProductDetail;

        const bool pads = padsRows( product.rows % kSmallTileRows, product.inputs, product.outputs, product.fewest_multiplies_to_pad );
        const int columns_in_tiles = product.rows < kSmallTileRows && !pads ? 0 : outputsInTiles( product.outputs );

        int next_row = 0;

        if ( columns_in_tiles > 0 )
        {
            next_row = encodeTiles<TPrecision, kLargeTileRows, kLargeTileSimdgroups>( context, pipelines.large_tiles, product, next_row, columns_in_tiles );
            next_row = encodeTiles<TPrecision, kSmallTileRows, kSmallTileSimdgroups>( context, pipelines.small_tiles, product, next_row, columns_in_tiles );
        }

        if ( pads )
        {
            encodePaddedRows<TPrecision>( context, pipelines, product, next_row, columns_in_tiles );
            next_row = product.rows;
        }

        // The rows no tile took, across every output, then the outputs no tile took on the rows tiles did.
        encodeOwn<TPrecision>( context, pipelines, product, next_row, product.rows, 0 );
        encodeOwn<TPrecision>( context, pipelines, product, 0, next_row, columns_in_tiles );
    }
}
