/**
 * @file MetalLinearOp.ixx
 * @brief A Linear's matrix multiply on Metal, on unquantized weights or on FP8, FP4, Q4_0, six-bit and codebook codes with their scales.
 */

export module Compute.MetalLinearOp;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalMatrixProduct;
import Compute.MetalMatrixView;
import Compute.MetalMemoryResource;
import Compute.MetalTensorDataTypeMap;
import Compute.OperationBase;
import Compute.OperationType;
import Dnn.Component;
import Dnn.Components.LinearConfig;
import Dnn.ITensor;
import Dnn.Quantization.Weight.Policies;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Serialization.Tensor;

namespace Mila::Dnn::Compute::Metal::Linear
{
    using namespace Mila::Dnn::Quant::Weight;

    /// One rectangle of a product over quantized weights, as `LinearQuantized.metal` reads it.
    struct QuantizedLinearShape
    {
        std::uint32_t inputs;
        std::uint32_t output_stride;
        std::uint32_t first_row;
        std::uint32_t rows;
        std::uint32_t first_output;
        std::uint32_t last_output;
        std::uint32_t row_bytes;
        std::uint32_t groups;
        std::uint32_t group;
    };

    struct OutputScaleShape
    {
        std::uint32_t columns;
        std::uint32_t output_stride;
    };

    struct WeightQuantizationShape
    {
        std::uint32_t inputs;
    };

    /**
     * @brief output = input * weight^T + bias.
     *
     * Quantized weights are FP8 codes with a scale per output, or FP4, Q4_0, six-bit or codebook codes with a scale per group of
     * inputs; a codebook's two- or three-bit codes index the tensor's own FP32 table.
     * The path each takes for a number of rows is the one `MetalBackend.md` section 4.2 measured: own lanes for a few rows; for
     * FP8, TensorOps on the bare codes and each output's scale after; for more rows, the weights expanded to the compute
     * precision a block of outputs at a time for TensorOps.
     *
     * @tparam TComputePrecision FP16 or BF16.
     * @tparam TWeightQuantization NoWeightQuant, PerChannelFp8<>, PerGroupFp4 of 64 or 128, PerGroupInt4<32>, PerGroupInt6<32>,
     *         PerGroupCodebook2 of 32 or 64, or PerGroupCodebook3 of 64 or 128.
     */
    export template<TensorDataType TComputePrecision, WeightQuantPolicy TWeightQuantization = NoWeightQuant>
    class MetalLinearOp : public Operation<DeviceType::Metal, TComputePrecision>
    {
    public:

        using TensorType = Tensor<TComputePrecision, MetalMemoryResource>;

        static constexpr bool kIsQuantized = TWeightQuantization::kIsQuantized;
        static constexpr bool kIsFp8Weight = kIsQuantized && TWeightQuantization::kPerChannel;
        static constexpr bool kIsFp4Weight = HasFp4E2M1Codes<TWeightQuantization>;
        static constexpr bool kIsInt4Weight = HasInt4Codes<TWeightQuantization>;
        static constexpr bool kIsInt6Weight = HasInt6Codes<TWeightQuantization>;
        static constexpr bool kIsCodebookWeight = HasCodebookTable<TWeightQuantization>;
        static constexpr bool kHasHighBitPlane = HasHighBitPlane<TWeightQuantization>;

        /// A scale for each group of inputs, rather than one for each output.
        static constexpr bool kIsGrouped = kIsFp4Weight || kIsInt4Weight || kIsInt6Weight || kIsCodebookWeight;

        static_assert( !kIsQuantized || kIsFp8Weight || kIsGrouped,
            "Metal's Linear carries FP8, FP4, Q4_0, six-bit and two- and three-bit codebook weights" );
        static_assert( !kIsFp8Weight || TWeightQuantization::kStorageDtype == TensorDataType::FP8_E4M3,
            "Metal's FP8 weights are E4M3" );

        MetalLinearOp( IExecutionContext* context, const LinearConfig& config )
            : context_( validateExecutionContext_<DeviceType::Metal>( context, "MetalLinearOp" ) ),
              config_( config ),
              pipelines_( resolvePipelines( context_->device().kernels(), config_.hasBias() ) )
        {
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            weight_ = weight;
            bias_ = bias;
        }

        /// Bind the scales: produced by `quantize()`, or loaded with weights stored quantized. One per output for FP8 and one per group
        /// otherwise; FP16 for Q4_0, six bits and the codebooks, FP32 for the rest, as each policy stores them.
        void setWeightScales( ITensor* scales ) requires kIsQuantized
        {
            if ( scales == nullptr || scales->getDeviceType() != DeviceType::Metal )
            {
                throw std::invalid_argument( "MetalLinearOp::setWeightScales - the scales must be a Metal tensor" );
            }

            weight_scales_ = scales;
        }

        /// Bind a codebook format's FP32 table and, for three bits, its plane of high bits. Borrowed, as the weights are.
        void setCodebookTensors( ITensor* codebook, ITensor* high_plane ) requires kIsCodebookWeight
        {
            if ( codebook == nullptr || codebook->getDeviceType() != DeviceType::Metal )
            {
                throw std::invalid_argument( "MetalLinearOp::setCodebookTensors - the codebook must be a Metal tensor" );
            }

            if ( kHasHighBitPlane != ( high_plane != nullptr ) )
            {
                throw std::invalid_argument( std::format( "MetalLinearOp::setCodebookTensors - {}-bit codes {} a plane of high bits",
                    TWeightQuantization::kCodeBits, kHasHighBitPlane ? "need" : "take no" ) );
            }

            codebook_ = codebook;
            high_plane_ = high_plane;
        }

        /// Nothing is derived from the scales here, so weights stored quantized need nothing more once they and their scales have landed.
        void onQuantizedWeightsLoaded() requires kIsQuantized
        {
        }

        /**
         * @brief Quantize BF16 weights, as a load hands them, into this format's codes and scales.
         *
         * Writes the bytes Mila's CUDA quantize writes for the same tensor. The weights go through the load's own staging buffer in
         * blocks of rows under its limit, and the work has finished when this returns.
         *
         * @throws std::invalid_argument When the weights are not BF16 of the expected shape, or any is not finite, naming the tensor.
         */
        void quantize( const Serialization::ITensorBlob& blob, ITensor& weight_out, ITensor& scales_out, const shape_t& expected_shape )
            requires kIsQuantized
        {
            if constexpr ( kIsInt4Weight || kIsInt6Weight || kIsCodebookWeight )
            {
                throw std::invalid_argument( std::format( "MetalLinearOp: '{}' holds full-precision weights, and Metal loads {} weights "
                    "only as stored in that format", weight_out.getName(), kFormatName ) );
            }

            if ( blob.getMetadata().dtype != TensorDataType::BF16 )
            {
                throw std::invalid_argument( std::format( "MetalLinearOp: '{}' is quantized from BF16 weights, and these are {}",
                    weight_out.getName(), tensorDataTypeToString( blob.getMetadata().dtype ) ) );
            }

            if ( blob.getMetadata().shape != expected_shape )
            {
                throw std::invalid_argument( std::format( "MetalLinearOp: '{}' expects weights of shape {}, and these are {}",
                    weight_out.getName(), shapeToString( expected_shape ), shapeToString( blob.getMetadata().shape ) ) );
            }

            if constexpr ( kIsFp8Weight || kIsFp4Weight )
            {
                quantizeRows( static_cast<const std::byte*>( blob.data() ), static_cast<std::size_t>( expected_shape[0] ),
                    static_cast<std::size_t>( expected_shape[1] ), weight_out, scales_out );
            }
        }

        /**
         * @brief `quantize()` past its checks: `outputs` rows of `inputs` BF16 weights on the host, each a row of this format.
         *
         * A stack of rows of one format, as a mixture-of-experts bank holds, quantizes as one Linear's weights.
         *
         * @throws std::invalid_argument When any weight is not finite, naming `weight_out`.
         */
        void quantizeRows( const std::byte* weights, std::size_t outputs, std::size_t inputs, ITensor& weight_out, ITensor& scales_out )
            requires ( kIsFp8Weight || kIsFp4Weight )
        {
            const std::size_t source_row_bytes = inputs * sizeof( std::uint16_t );
            const std::size_t rows_per_block = std::max<std::size_t>( 1, MetalExecutionContext::getLoadStagingLimitBytes() / source_row_bytes );

            // The count of weights that are not finite sits ahead of the weights, a whole binding's alignment away.
            constexpr std::size_t kCountBytes = 256;

            context_->synchronize();

            auto* const staging = static_cast<std::byte*>(
                context_->getLoadStagingBuffer( kCountBytes + std::min( outputs, rows_per_block ) * source_row_bytes ) );
            auto* const not_finite = reinterpret_cast<std::uint32_t*>( staging );
            std::byte* const source = staging + kCountBytes;

            *not_finite = 0;

            auto* const codes = static_cast<std::byte*>( weight_out.rawData() );
            auto* const scales = static_cast<float*>( scales_out.rawData() );
            const WeightQuantizationShape shape{ static_cast<std::uint32_t>( inputs ) };

            for ( std::size_t first = 0; first < outputs; first += rows_per_block )
            {
                const std::size_t rows = std::min( rows_per_block, outputs - first );

                // The host writes this buffer, so the block before must have been read.
                context_->synchronize();
                std::memcpy( source, weights + first * source_row_bytes, rows * source_row_bytes );

                const std::array<const void*, 4> buffers{
                    source, codes + first * codeBytesPerOutput( inputs ), scales + first * scalesPerOutput( inputs ), not_finite };

                if constexpr ( kIsFp8Weight )
                {
                    context_->dispatchThreadgroups( pipelines_.quantize, buffers, std::as_bytes( std::span( &shape, 1 ) ), rows, 1, kFp8QuantizationSimdgroups );
                } else {
                    context_->dispatchThreadgroups(
                        pipelines_.quantize, buffers, std::as_bytes( std::span( &shape, 1 ) ), inputs / kGroup, rows, kGroup / kSimdgroupWidth );
                }
            }

            context_->synchronize();

            if ( *not_finite != 0 )
            {
                throw std::invalid_argument( std::format( "MetalLinearOp: {} of the weights of '{}' are not finite, and cannot be quantized",
                    *not_finite, weight_out.getName() ) );
            }
        }

        void build( const BuildContext& build_context ) override
        {
            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "MetalLinearOp::build - setParameters() must be called before build()" );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "MetalLinearOp::build - bias expected by config but not bound" );
            }

            const shape_t& input_shape = build_context.inputShape();

            if ( input_shape.empty() || input_shape.back() != config_.getInputFeatures() )
            {
                throw std::invalid_argument( "MetalLinearOp::build - the input's last axis must be the configured input features" );
            }

            inputs_ = narrowToKernelIndex( config_.getInputFeatures() );
            outputs_ = narrowToKernelIndex( config_.getOutputFeatures() );

            if constexpr ( kIsQuantized )
            {
                if ( weight_scales_ == nullptr )
                {
                    throw std::runtime_error( "MetalLinearOp::build - setWeightScales() must be called before build()" );
                }

                if constexpr ( kIsCodebookWeight )
                {
                    if ( codebook_ == nullptr )
                    {
                        throw std::runtime_error( "MetalLinearOp::build - setCodebookTensors() must be called before build()" );
                    }
                }

                // Eight weights are read at a time, from codes that start on four bytes. A group of 32 also keeps a six-bit row, three
                // quarters of a byte an input, on whole words.
                const int multiple = std::max( kChunk, kIsGrouped ? kGroupOf( inputs_ ) : kChunk );

                if ( inputs_ % multiple != 0 )
                {
                    throw std::invalid_argument( std::format(
                        "MetalLinearOp::build - quantized weights need input features a multiple of {}, and these are {}", multiple, inputs_ ) );
                }
            }

            scratch_bytes_ = scratchBytesFor( input_shape );

            Operation<DeviceType::Metal, TComputePrecision>::build( build_context );
        }

        /// Encode the multiply. The rows are however many the input holds this call, and `output` holds at least that many.
        void forward( const TensorType& input, TensorType& output ) const
        {
            const int rows = narrowToKernelIndex( input.size() / config_.getInputFeatures() );
            const void* const input_address = static_cast<const ITensor&>( input ).rawData();
            const void* const output_address = static_cast<const ITensor&>( output ).rawData();

            if constexpr ( kIsQuantized )
            {
                encodeQuantized( input_address, output_address, rows );
            } else {
                encodeMatrixProduct<TComputePrecision>( *context_, pipelines_.product, MatrixProduct{
                    input_address, static_cast<const ITensor*>( weight_ )->rawData(), output_address,
                    rows, inputs_, outputs_, outputs_, kFewestMultipliesToPad } );
            }

            if ( config_.hasBias() )
            {
                encodeBias( output_address, rows );
            }
        }

        /**
         * @brief Decode's arithmetic for each of up to kDecodeRowsMaximum rows, every weight read once for all of them (`MetalBackend.md`
         *        section 4.6, `Gemma4Mtp.md` 4.7).
         *
         * Quantized weights take own lanes at every row count, where forward() would move to TensorOps from more rows; one row keeps
         * its own kernel, so a row of several equals a one-row decode to FP32 rounding, not to the bit. Unquantized weights take
         * forward()'s matrix product.
         *
         * @throws std::invalid_argument for more than kDecodeRowsMaximum rows.
         */
        void decode( const TensorType& input, TensorType& output ) const
        {
            const int rows = narrowToKernelIndex( input.size() / config_.getInputFeatures() );

            if ( rows > kDecodeRowsMaximum )
            {
                throw std::invalid_argument( std::format( "MetalLinearOp::decode: {} rows; decode takes at most {}", rows, kDecodeRowsMaximum ) );
            }

            const void* const input_address = static_cast<const ITensor&>( input ).rawData();
            const void* const output_address = static_cast<const ITensor&>( output ).rawData();

            if constexpr ( kIsQuantized )
            {
                encodeLanes( input_address, output_address, 0, rows, 0, outputs_ );
            } else {
                encodeMatrixProduct<TComputePrecision>( *context_, pipelines_.product, MatrixProduct{
                    input_address, static_cast<const ITensor*>( weight_ )->rawData(), output_address,
                    rows, inputs_, outputs_, outputs_, kFewestMultipliesToPad } );
            }

            if ( config_.hasBias() )
            {
                encodeBias( output_address, rows );
            }
        }

        /// The most rows decode() takes: a speculative round's draft and the token before it.
        static constexpr int kDecodeRowsMaximum = 8;

        [[nodiscard]] OperationType getOperationType() const override
        {
            return OperationType::LinearOp;
        }

        [[nodiscard]] std::string getName() const override
        {
            return "Metal.Linear";
        }

        [[nodiscard]] std::size_t getScratchBytes() const override
        {
            return scratch_bytes_;
        }

        [[nodiscard]] std::size_t getRequiredScratchBytes( const BuildContext& build_context ) const override
        {
            return scratchBytesFor( build_context.inputShape() );
        }

    private:

        // Measured on the M1 Pro at Linear's shapes, in FP16: from here up padding won every shape by more than Phase 2's 10%, and below
        // it one shape won by only 3%. Notebooks/MetalLlamaOperations.md has the table.
        static constexpr std::int64_t kFewestMultipliesToPad = 50'000'000;

        static constexpr int kChunk = 8;
        static constexpr int kTileRows = 8;
        static constexpr int kLargeTileRows = 128;
        static constexpr int kTileColumns = 16;
        static constexpr int kSimdgroupWidth = 32;
        static constexpr int kFp8QuantizationSimdgroups = 8;
        static constexpr int kExpandBlockOutputs = 1024;

        // TensorOps multiplies FP16 activations by FP8's bare codes, and refuses BF16 ones, so BF16 takes FP4's paths.
        static constexpr bool kMultipliesBareCodes = kIsFp8Weight && TComputePrecision == TensorDataType::FP16;

        // From these rows up, FP8 multiplies its bare codes by TensorOps, and a format multiplies weights expanded a block at a time,
        // from the first after bare codes and from the second after own lanes. Notebooks/MetalQuantizedWeights.md has the timing.
        static constexpr int kFewestRowsForBareCodes = 8;
        static constexpr int kBareCodesFewestRowsToExpand = 384;
        static constexpr int kLanesFewestRowsToExpand = 16;

        static constexpr int kGroup = [] {
            if constexpr ( kIsGrouped )
            {
                return TWeightQuantization::kQuantizationGroupSize;
            } else {
                return 0;
            }
        }();

        static constexpr int kGroupOf( int inputs ) noexcept
        {
            return kIsGrouped ? kGroup : inputs;
        }

        static constexpr std::size_t codeBytesPerOutput( std::size_t inputs ) noexcept
        {
            if constexpr ( kIsInt6Weight )
            {
                return inputs / 2 + inputs / 4;
            }
            else if constexpr ( kIsCodebookWeight )
            {
                // The two-bit plane; three bits' high plane is its own tensor.
                return inputs / 4;
            } else {
                return kIsFp4Weight || kIsInt4Weight ? inputs / 2 : inputs;
            }
        }

        static constexpr std::size_t scalesPerOutput( std::size_t inputs ) noexcept
        {
            if constexpr ( kIsGrouped )
            {
                return inputs / static_cast<std::size_t>( kGroup );
            } else {
                return 1;
            }
        }

        static constexpr std::string_view kFormatName = kIsCodebookWeight ? ( kHasHighBitPlane ? "codebook3" : "codebook2" )
            : kIsInt6Weight ? "int6" : kIsInt4Weight ? "int4" : kIsFp4Weight ? "fp4" : "fp8";

        static constexpr int kFewestRowsToExpand = kMultipliesBareCodes ? kBareCodesFewestRowsToExpand : kLanesFewestRowsToExpand;

        // One row of Q4_0 takes linear_quantized_row, MLX's one-row shape (MetalBackend.md section 4.2): eight outputs a
        // threadgroup, sixteen inputs a lane, which Q4_0's group of 32 never splits.
        static constexpr bool kTakesRowKernel = kIsInt4Weight;
        static constexpr int kRowKernelOutputs = 8;

        static_assert( !kTakesRowKernel || kGroup % 16 == 0, "linear_quantized_row reads sixteen inputs under one scale" );

        // Every kernel this op may dispatch, resolved once when it is constructed: a dispatch names none. One this format, precision
        // or configuration never reaches stays null.
        struct Pipelines
        {
            /// Unquantized weights multiply through these, and so do quantized ones once expanded.
            MatrixProductPipelines product;

            MetalPipeline bias{ nullptr };

            /// One row of Q4_0, and the own lanes: one row four outputs across, and up to eight rows together.
            MetalPipeline row{ nullptr };
            MetalPipeline one_row_lanes{ nullptr };
            MetalPipeline rows_lanes{ nullptr };

            MetalPipeline expand{ nullptr };

            /// FP8's bare codes through TensorOps, in tiles of 128 rows and of 8, and each output's scale after.
            MetalPipeline large_fp8_tiles{ nullptr };
            MetalPipeline small_fp8_tiles{ nullptr };
            MetalPipeline output_scales{ nullptr };

            /// A load's full-precision weights into this format.
            MetalPipeline quantize{ nullptr };
        };

        [[nodiscard]] static Pipelines resolvePipelines( MetalKernelLibrary& kernels, bool has_bias )
        {
            const std::string_view type = MetalTensorDataTypeMap<TComputePrecision>::kernel_type;

            Pipelines pipelines;
            pipelines.product = matrixProductPipelines<TComputePrecision>( kernels );

            if ( has_bias )
            {
                pipelines.bias = kernels.pipeline( std::format( "linear_add_bias_{}", type ) );
            }

            if constexpr ( kIsQuantized )
            {
                const std::string_view family = kIsCodebookWeight ? "codebook" : "quantized";

                // One row of a format with a row kernel never reaches the lanes, which have no one-row instance for it.
                if constexpr ( kTakesRowKernel )
                {
                    pipelines.row = kernels.pipeline( std::format( "linear_quantized_row_{}_{}", kFormatName, type ) );
                } else {
                    pipelines.one_row_lanes = kernels.pipeline( std::format( "linear_{}_lanes_{}_{}_4_1", family, kFormatName, type ) );
                }

                pipelines.rows_lanes = kernels.pipeline( std::format( "linear_{}_lanes_{}_{}_1_8", family, kFormatName, type ) );
                pipelines.expand = kernels.pipeline( std::format( "linear_{}_expand_{}_{}", family, kFormatName, type ) );
            }

            if constexpr ( kMultipliesBareCodes )
            {
                pipelines.large_fp8_tiles = kernels.pipeline( std::format( "linear_fp8_tiles_{}_{}_{}_{}", type, kLargeTileRows, kTileColumns, 4 ) );
                pipelines.small_fp8_tiles = kernels.pipeline( std::format( "linear_fp8_tiles_{}_{}_{}_{}", type, kTileRows, kTileColumns, 2 ) );
                pipelines.output_scales = kernels.pipeline( std::format( "linear_apply_output_scales_{}", type ) );
            }

            if constexpr ( kIsFp8Weight )
            {
                pipelines.quantize = kernels.pipeline( "linear_quantize_fp8_per_channel" );
            }
            else if constexpr ( kIsFp4Weight )
            {
                pipelines.quantize = kernels.pipeline( std::format( "linear_quantize_fp4_per_group_{}", kGroup ) );
            }

            return pipelines;
        }

        [[nodiscard]] static constexpr int outputsInTiles( int outputs ) noexcept
        {
            return outputs - outputs % kTileColumns;
        }

        [[nodiscard]] std::size_t scratchBytesFor( const shape_t& input_shape ) const
        {
            const int rows = narrowToKernelIndex( elementCount( input_shape ) / config_.getInputFeatures() );
            const int inputs = narrowToKernelIndex( config_.getInputFeatures() );
            const int outputs = narrowToKernelIndex( config_.getOutputFeatures() );

            if constexpr ( !kIsQuantized )
            {
                // A Linear built for one row, as a generating model's output layer is, never holds rows to pad.
                return matrixProductScratchBytes<TComputePrecision>( rows, inputs, outputs, kFewestMultipliesToPad );
            } else {
                std::size_t bytes = 0;

                if ( kMultipliesBareCodes && rows >= kFewestRowsForBareCodes )
                {
                    const int most_rows = std::min( rows, kFewestRowsToExpand - 1 );
                    const int rows_in_tiles = most_rows - most_rows % kTileRows;

                    bytes = sizeof( float ) * static_cast<std::size_t>( rows_in_tiles ) * static_cast<std::size_t>( outputsInTiles( outputs ) );
                }

                if ( rows >= kFewestRowsToExpand )
                {
                    bytes = std::max( bytes, metalElementBytes<TComputePrecision>(
                        static_cast<dim_t>( std::min( kExpandBlockOutputs, outputs ) ) * inputs ) );
                }

                return bytes;
            }
        }

        [[nodiscard]] QuantizedLinearShape shapeOf( int first_row, int rows, int first_output, int last_output ) const
        {
            return QuantizedLinearShape{
                static_cast<std::uint32_t>( inputs_ ), static_cast<std::uint32_t>( outputs_ ),
                static_cast<std::uint32_t>( first_row ), static_cast<std::uint32_t>( rows ),
                static_cast<std::uint32_t>( first_output ), static_cast<std::uint32_t>( last_output ),
                static_cast<std::uint32_t>( codeBytesPerOutput( static_cast<std::size_t>( inputs_ ) ) ),
                static_cast<std::uint32_t>( scalesPerOutput( static_cast<std::size_t>( inputs_ ) ) ),
                static_cast<std::uint32_t>( kGroupOf( inputs_ ) ) };
        }

        void encodeQuantized( const void* input, const void* output, int rows ) const
        {
            if ( rows >= kFewestRowsToExpand )
            {
                encodeExpanded( input, output, rows );
            }
            else if ( kMultipliesBareCodes && rows >= kFewestRowsForBareCodes )
            {
                encodeBareCodes( input, output, rows );
            } else {
                encodeLanes( input, output, 0, rows, 0, outputs_ );
            }
        }

        /// Own lanes over rows [first_row, first_row + rows) by outputs [first_output, last_output).
        void encodeLanes( const void* input, const void* output, int first_row, int rows, int first_output, int last_output ) const
        {
            if ( rows <= 0 || last_output <= first_output )
            {
                return;
            }

            const QuantizedLinearShape shape = shapeOf( first_row, rows, first_output, last_output );
            const void* const weight = static_cast<const ITensor*>( weight_ )->rawData();
            const void* const scales = static_cast<const ITensor*>( weight_scales_ )->rawData();

            if constexpr ( kTakesRowKernel )
            {
                if ( rows == 1 )
                {
                    // Two SIMD-groups of four outputs a threadgroup: LinearQuantized.metal, linear_quantized_row.
                    const std::array<const void*, 4> buffers{ input, weight, scales, output };
                    const std::size_t width = static_cast<std::size_t>( ( last_output - first_output + kRowKernelOutputs - 1 ) / kRowKernelOutputs );

                    context_->dispatchThreadgroups( pipelines_.row, buffers, std::as_bytes( std::span( &shape, 1 ) ), width, 1, 2 );

                    return;
                }
            }

            const int across = rows == 1 ? 4 : 1;
            const int rows_together = rows == 1 ? 1 : 8;
            const MetalPipeline pipeline = rows == 1 ? pipelines_.one_row_lanes : pipelines_.rows_lanes;
            const std::size_t width = static_cast<std::size_t>( ( last_output - first_output + across - 1 ) / across );
            const std::size_t height = static_cast<std::size_t>( ( rows + rows_together - 1 ) / rows_together );

            if constexpr ( kIsCodebookWeight )
            {
                const std::array<const void*, 6> buffers{ input, weight, scales, output, codebookAddress(), highPlaneAddress() };

                context_->dispatchThreadgroups( pipeline, buffers, std::as_bytes( std::span( &shape, 1 ) ), width, height );
            } else {
                const std::array<const void*, 4> buffers{ input, weight, scales, output };

                context_->dispatchThreadgroups( pipeline, buffers, std::as_bytes( std::span( &shape, 1 ) ), width, height );
            }
        }

        /// FP8: TensorOps on the bare codes into 32-bit sums over whole tiles, each output's scale after, and own lanes for the rest.
        void encodeBareCodes( const void* input, const void* output, int rows ) const
        {
            const int rows_in_tiles = rows - rows % kTileRows;
            const int columns = outputsInTiles( outputs_ );

            if ( columns > 0 )
            {
                auto* const sums = static_cast<float*>( context_->getDeviceScratchBuffer(
                    sizeof( float ) * static_cast<std::size_t>( rows_in_tiles ) * static_cast<std::size_t>( columns ) ) );

                int next_row = encodeFp8Tiles<kLargeTileRows, 4>( pipelines_.large_fp8_tiles, input, sums, 0, rows_in_tiles, columns );
                next_row = encodeFp8Tiles<kTileRows, 2>( pipelines_.small_fp8_tiles, input, sums, next_row, rows_in_tiles, columns );

                const OutputScaleShape shape{ static_cast<std::uint32_t>( columns ), static_cast<std::uint32_t>( outputs_ ) };
                const std::array<const void*, 3> buffers{ sums, static_cast<const ITensor*>( weight_scales_ )->rawData(), output };

                context_->dispatch(
                    pipelines_.output_scales, buffers, std::as_bytes( std::span( &shape, 1 ) ),
                    static_cast<std::size_t>( rows_in_tiles ) * static_cast<std::size_t>( columns ) );
            }

            encodeLanes( input, output, 0, rows_in_tiles, columns, outputs_ );
            encodeLanes( input, output, rows_in_tiles, rows - rows_in_tiles, 0, outputs_ );
        }

        /// Tiles of TTileRows from `first_row` up to `last_row`, returning the first row they leave.
        template<int TTileRows, int TSimdgroups>
        int encodeFp8Tiles( MetalPipeline pipeline, const void* input, float* sums, int first_row, int last_row, int columns ) const
        {
            const int tiles_down = ( last_row - first_row ) / TTileRows;

            if ( tiles_down == 0 )
            {
                return first_row;
            }

            const int rows_taken = tiles_down * TTileRows;
            const auto* const first_input = static_cast<const std::byte*>( input )
                + metalElementBytes<TComputePrecision>( static_cast<dim_t>( first_row ) * inputs_ );

            const std::array<MetalMatrixView, 3> matrices{
                MetalMatrixView{ first_input, TComputePrecision, rows_taken, inputs_, inputs_ },
                MetalMatrixView{ static_cast<const ITensor*>( weight_ )->rawData(), TensorDataType::FP8_E4M3, columns, inputs_, inputs_ },
                MetalMatrixView{ sums + static_cast<std::size_t>( first_row ) * static_cast<std::size_t>( columns ), TensorDataType::FP32,
                    rows_taken, columns, columns } };

            context_->dispatchMatrices(
                pipeline, matrices, static_cast<std::size_t>( columns / kTileColumns ), static_cast<std::size_t>( tiles_down ), TSimdgroups );

            return first_row + rows_taken;
        }

        /// The weights expanded to the compute precision a block of outputs at a time, each block multiplied as unquantized weights are.
        void encodeExpanded( const void* input, const void* output, int rows ) const
        {
            const int block = std::min( kExpandBlockOutputs, outputs_ );
            void* const expanded = context_->getDeviceScratchBuffer(
                metalElementBytes<TComputePrecision>( static_cast<dim_t>( block ) * inputs_ ) );

            const void* const weight = static_cast<const ITensor*>( weight_ )->rawData();
            const void* const scales = static_cast<const ITensor*>( weight_scales_ )->rawData();

            for ( int first = 0; first < outputs_; first += block )
            {
                const int count = std::min( block, outputs_ - first );
                const QuantizedLinearShape shape = shapeOf( 0, rows, first, first + count );
                const std::size_t threads = static_cast<std::size_t>( count ) * static_cast<std::size_t>( inputs_ / kChunk );

                if constexpr ( kIsCodebookWeight )
                {
                    const std::array<const void*, 5> buffers{ weight, scales, expanded, codebookAddress(), highPlaneAddress() };

                    context_->dispatch( pipelines_.expand, buffers, std::as_bytes( std::span( &shape, 1 ) ), threads );
                } else {
                    const std::array<const void*, 3> buffers{ weight, scales, expanded };

                    context_->dispatch( pipelines_.expand, buffers, std::as_bytes( std::span( &shape, 1 ) ), threads );
                }

                const auto* const block_output = static_cast<const std::byte*>( output ) + metalElementBytes<TComputePrecision>( first );

                encodeMatrixProduct<TComputePrecision>( *context_, pipelines_.product, MatrixProduct{
                    input, expanded, block_output, rows, inputs_, count, outputs_, kNeverPad } );
            }
        }

        [[nodiscard]] const void* codebookAddress() const
        {
            return static_cast<const ITensor*>( codebook_ )->rawData();
        }

        /// Two bits have no high plane, and the kernel never reads one: the codes stand in for its binding.
        [[nodiscard]] const void* highPlaneAddress() const
        {
            return high_plane_ ? static_cast<const ITensor*>( high_plane_ )->rawData() : static_cast<const ITensor*>( weight_ )->rawData();
        }

        void encodeBias( const void* output, int rows ) const
        {
            const std::array<const void*, 2> buffers{ static_cast<const ITensor*>( bias_ )->rawData(), output };
            const std::uint32_t outputs = static_cast<std::uint32_t>( outputs_ );

            context_->dispatch(
                pipelines_.bias, buffers, std::as_bytes( std::span( &outputs, 1 ) ),
                static_cast<std::size_t>( rows ) * static_cast<std::size_t>( outputs_ ) );
        }

        MetalExecutionContext* context_;
        LinearConfig config_;
        Pipelines pipelines_;
        ITensor* weight_{ nullptr };
        ITensor* bias_{ nullptr };
        ITensor* weight_scales_{ nullptr };
        ITensor* codebook_{ nullptr };
        ITensor* high_plane_{ nullptr };
        int inputs_{ 0 };
        int outputs_{ 0 };
        std::size_t scratch_bytes_{ 0 };
    };
}
