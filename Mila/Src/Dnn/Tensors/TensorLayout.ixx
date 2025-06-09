/**
 * @file TensorLayout.ixx
 * @brief Abstract interface for tensor layouts used in computational operations
 *
 * Provides a backend-agnostic abstraction of tensor memory layouts for use with
 * various accelerators like CUDA, cuBLASLt, etc. The layout describes how tensor
 * data is arranged in memory, including dimensions, strides, and memory format.
 */

module;
#include <vector>
#include <cstddef>
#include <memory>
#include <string>
#include <stdexcept>
#include <cublasLt.h>

export module Dnn.TensorLayout;

import Dnn.TensorTraits;

namespace Mila::Dnn
{
    /**
     * @brief Enumeration of supported tensor memory formats
     */
    export enum class MemoryFormat {
        RowMajor,      // Standard C/C++ row-major layout (default)
        ColumnMajor,   // Column-major layout (used by FORTRAN, BLAS, cuBLAS)
        ChannelsLast,  // NHWC format for vision models (N, H, W, C)
        ChannelsFirst  // NCHW format for vision models (N, C, H, W)
    };

    /**
     * @brief Abstract base class defining tensor layout interface
     *
     * TensorLayout represents how tensor data is arranged in memory,
     * abstracting implementation details from platform-specific backends.
     * Concrete implementations will provide adapter functionality to
     * different accelerator libraries like cuBLASLt, MIOpen, etc.
     */
    export class TensorLayout {
    public:
        /**
         * @brief Virtual destructor for proper cleanup in derived classes
         */
        virtual ~TensorLayout() = default;

        /**
         * @brief Get the number of dimensions in the tensor
         * @return Number of dimensions
         */
        virtual size_t rank() const = 0;

        /**
         * @brief Get the tensor shape (dimensions)
         * @return Vector containing the size of each dimension
         */
        virtual const std::vector<size_t>& shape() const = 0;

        /**
         * @brief Get the strides for each dimension
         * @return Vector containing the stride (step size) for each dimension
         */
        virtual const std::vector<size_t>& strides() const = 0;

        /**
         * @brief Get the total number of elements in the tensor
         * @return Total element count
         */
        virtual size_t size() const = 0;

        /**
         * @brief Get the memory format of the tensor
         * @return Format specification
         */
        virtual MemoryFormat format() const = 0;

        /**
         * @brief Get the leading dimension for matrix operations
         *
         * For matrices, this returns the memory distance between consecutive
         * rows (for row-major) or columns (for column-major).
         * This is especially important for BLAS-like operations.
         *
         * @return The leading dimension value
         */
        virtual size_t leadingDimension() const = 0;

        /**
         * @brief Calculate the linear memory offset for the given indices
         *
         * @param indices Array of indices along each dimension
         * @return The memory offset in elements (not bytes)
         * @throws std::out_of_range if indices are invalid
         */
        // FUTURE: virtual size_t offset( const std::vector<size_t>& indices ) const = 0;

        /**
         * @brief Check if the layout is contiguous in memory
         *
         * A layout is contiguous if elements are stored in a single unbroken
         * sequence with no holes or padding.
         *
         * @return True if contiguous, false otherwise
         */
        virtual bool isContiguous() const = 0;

        /**
         * @brief Create a new layout with different dimensions but same format
         *
         * @param newShape The desired shape for the new layout
         * @return Unique pointer to new TensorLayout object
         * @throws std::invalid_argument if shape is invalid
         */
        // FUTURE: virtual std::unique_ptr<TensorLayout> reshape( const std::vector<size_t>& newShape ) const = 0;

        /**
         * @brief Create a new layout with transposed dimensions
         *
         * For rank-2 tensors, this swaps rows and columns.
         * For higher-rank tensors, specific implementations define behavior.
         *
         * @return Unique pointer to new transposed TensorLayout
         */
        // FUTURE: virtual std::unique_ptr<TensorLayout> transpose() const = 0;
    };

    /**
     * @brief Standard tensor layout implementation with row-major storage
     *
     * This class provides a common implementation of TensorLayout with
     * row-major memory layout (C/C++ standard), suitable for most
     * tensor operations within the framework.
     */
    //export class StandardTensorLayout : public TensorLayout {
    //public:
    //    /**
    //     * @brief Construct a new tensor layout with the given shape
    //     *
    //     * @param dimensions Vector specifying tensor dimensions
    //     * @param format Memory layout format (defaults to RowMajor)
    //     */
    //    explicit StandardTensorLayout(
    //        const std::vector<size_t>& dimensions,
    //        MemoryFormat format = MemoryFormat::RowMajor )
    //        : shape_( dimensions ),
    //        format_( format ) {

    //        if ( dimensions.empty() ) {
    //            throw std::invalid_argument( "Tensor dimensions cannot be empty" );
    //        }

    //        computeStrides();
    //        computeSize();
    //    }

    //    /**
    //     * @brief Get the number of dimensions
    //     * @return Number of dimensions (rank)
    //     */
    //    size_t rank() const override {
    //        return shape_.size();
    //    }

    //    /**
    //     * @brief Get the shape vector
    //     * @return Constant reference to the shape vector
    //     */
    //    const std::vector<size_t>& shape() const override {
    //        return shape_;
    //    }

    //    /**
    //     * @brief Get the strides vector
    //     * @return Constant reference to the strides vector
    //     */
    //    const std::vector<size_t>& strides() const override {
    //        return strides_;
    //    }

    //    /**
    //     * @brief Get the total number of elements
    //     * @return Total element count
    //     */
    //    size_t size() const override {
    //        return size_;
    //    }

    //    /**
    //     * @brief Get the memory format
    //     * @return Memory format enum value
    //     */
    //    MemoryFormat format() const override {
    //        return format_;
    //    }

    //    /**
    //     * @brief Get the leading dimension for matrix operations
    //     *
    //     * For row-major matrices, this is the number of columns.
    //     * For column-major matrices, this is the number of rows.
    //     *
    //     * @return Leading dimension value
    //     */
    //    size_t leadingDimension() const override {
    //        if ( rank() < 2 ) {
    //            return 1; // For vectors or scalars
    //        }

    //        if ( format_ == MemoryFormat::RowMajor ) {
    //            // For row-major, leading dimension is the stride between rows
    //            return shape_[ rank() - 1 ];
    //        }
    //        else if ( format_ == MemoryFormat::ColumnMajor ) {
    //            // For column-major, leading dimension is the stride between columns
    //            return shape_[ 0 ];
    //        }
    //        else {
    //            // For other formats, return appropriate value based on memory layout
    //            return strides_[ 0 ];
    //        }
    //    }

    //    /**
    //     * @brief Calculate memory offset for the given indices
    //     *
    //     * @param indices Array of indices along each dimension
    //     * @return Linear memory offset in elements
    //     * @throws std::out_of_range if indices are invalid
    //     */
    //    size_t offset( const std::vector<size_t>& indices ) const override {
    //        if ( indices.size() != rank() ) {
    //            throw std::invalid_argument( "Number of indices must match tensor rank" );
    //        }

    //        size_t linearIndex = 0;
    //        for ( size_t i = 0; i < indices.size(); ++i ) {
    //            if ( indices[ i ] >= shape_[ i ] ) {
    //                throw std::out_of_range( "Index out of bounds" );
    //            }
    //            linearIndex += indices[ i ] * strides_[ i ];
    //        }

    //        return linearIndex;
    //    }

    //    /**
    //     * @brief Check if the layout is contiguous in memory
    //     * @return True if contiguous, false otherwise
    //     */
    //    bool isContiguous() const override {
    //        // Check if strides match computed contiguous strides
    //        std::vector<size_t> contiguousStrides = computeContiguousStrides( shape_, format_ );
    //        return strides_ == contiguousStrides;
    //    }

    //    /**
    //     * @brief Create a new layout with different dimensions
    //     *
    //     * @param newShape The desired shape for the new layout
    //     * @return Unique pointer to new TensorLayout
    //     * @throws std::invalid_argument if total size doesn't match
    //     */
    //    std::unique_ptr<TensorLayout> reshape( const std::vector<size_t>& newShape ) const override {
    //        size_t newSize = 1;
    //        for ( auto dim : newShape ) {
    //            newSize *= dim;
    //        }

    //        if ( newSize != size_ ) {
    //            throw std::invalid_argument( "Reshape must preserve total element count" );
    //        }

    //        return std::make_unique<StandardTensorLayout>( newShape, format_ );
    //    }

    //    /**
    //     * @brief Create a new layout with transposed dimensions
    //     *
    //     * For matrices, this swaps rows and columns.
    //     * For higher-rank tensors, this reverses the dimension order.
    //     *
    //     * @return Unique pointer to new transposed TensorLayout
    //     */
    //    std::unique_ptr<TensorLayout> transpose() const override {
    //        if ( rank() < 2 ) {
    //            // No change for scalars or vectors
    //            return std::make_unique<StandardTensorLayout>( shape_, format_ );
    //        }

    //        // For matrices, swap dimensions
    //        if ( rank() == 2 ) {
    //            std::vector<size_t> transposedShape = { shape_[ 1 ], shape_[ 0 ] };

    //            // Invert the memory format if needed
    //            MemoryFormat newFormat = format_;
    //            if ( format_ == MemoryFormat::RowMajor ) {
    //                newFormat = MemoryFormat::ColumnMajor;
    //            }
    //            else if ( format_ == MemoryFormat::ColumnMajor ) {
    //                newFormat = MemoryFormat::RowMajor;
    //            }

    //            return std::make_unique<StandardTensorLayout>( transposedShape, newFormat );
    //        }

    //        // For higher-rank tensors, reverse all dimensions
    //        std::vector<size_t> transposedShape( shape_.rbegin(), shape_.rend() );
    //        return std::make_unique<StandardTensorLayout>( transposedShape, format_ );
    //    }

    //    /**
    //     * @brief Create string representation of the layout
    //     * @return String describing the layout
    //     */
    //    std::string toString() const override {
    //        std::string result = "TensorLayout(shape=[";

    //        for ( size_t i = 0; i < shape_.size(); ++i ) {
    //            result += std::to_string( shape_[ i ] );
    //            if ( i < shape_.size() - 1 ) result += ",";
    //        }

    //        result += "], strides=[";

    //        for ( size_t i = 0; i < strides_.size(); ++i ) {
    //            result += std::to_string( strides_[ i ] );
    //            if ( i < strides_.size() - 1 ) result += ",";
    //        }

    //        result += "], format=";

    //        switch ( format_ ) {
    //            case MemoryFormat::RowMajor:
    //                result += "RowMajor";
    //                break;
    //            case MemoryFormat::ColumnMajor:
    //                result += "ColumnMajor";
    //                break;
    //            case MemoryFormat::ChannelsLast:
    //                result += "ChannelsLast";
    //                break;
    //            case MemoryFormat::ChannelsFirst:
    //                result += "ChannelsFirst";
    //                break;
    //        }

    //        result += ", size=" + std::to_string( size_ ) + ")";
    //        return result;
    //    }

    //protected:
    //    std::vector<size_t> shape_;    ///< Tensor dimensions
    //    std::vector<size_t> strides_;  ///< Memory stride for each dimension
    //    size_t size_;                  ///< Total number of elements
    //    MemoryFormat format_;          ///< Memory layout format

    //    /**
    //     * @brief Compute memory strides based on shape and format
    //     */
    //    void computeStrides() {
    //        strides_.resize( shape_.size() );

    //        // Handle different memory layouts
    //        switch ( format_ ) {
    //            case MemoryFormat::RowMajor:
    //                computeRowMajorStrides();
    //                break;

    //            case MemoryFormat::ColumnMajor:
    //                computeColumnMajorStrides();
    //                break;

    //            case MemoryFormat::ChannelsLast:
    //                computeChannelsLastStrides();
    //                break;

    //            case MemoryFormat::ChannelsFirst:
    //                computeChannelsFirstStrides();
    //                break;
    //        }
    //    }

    //    /**
    //     * @brief Compute row-major strides (C-style)
    //     */
    //    void computeRowMajorStrides() {
    //        strides_[ strides_.size() - 1 ] = 1;
    //        for ( int i = strides_.size() - 2; i >= 0; --i ) {
    //            strides_[ i ] = strides_[ i + 1 ] * shape_[ i + 1 ];
    //        }
    //    }

    //    /**
    //     * @brief Compute column-major strides (Fortran-style)
    //     */
    //    void computeColumnMajorStrides() {
    //        strides_[ 0 ] = 1;
    //        for ( size_t i = 1; i < strides_.size(); ++i ) {
    //            strides_[ i ] = strides_[ i - 1 ] * shape_[ i - 1 ];
    //        }
    //    }

    //    /**
    //     * @brief Compute strides for NHWC format (channels-last)
    //     * Assumes 4D tensor with [batch, height, width, channels]
    //     */
    //    void computeChannelsLastStrides() {
    //        if ( shape_.size() != 4 ) {
    //            throw std::invalid_argument( "ChannelsLast format requires 4D tensor" );
    //        }

    //        // NHWC: channels vary fastest, then width, height, batch
    //        strides_[ 3 ] = 1;                     // C stride
    //        strides_[ 2 ] = shape_[ 3 ];             // W stride
    //        strides_[ 1 ] = shape_[ 3 ] * shape_[ 2 ]; // H stride
    //        strides_[ 0 ] = shape_[ 3 ] * shape_[ 2 ] * shape_[ 1 ]; // N stride
    //    }

    //    /**
    //     * @brief Compute strides for NCHW format (channels-first)
    //     * Assumes 4D tensor with [batch, channels, height, width]
    //     */
    //    void computeChannelsFirstStrides() {
    //        if ( shape_.size() != 4 ) {
    //            throw std::invalid_argument( "ChannelsFirst format requires 4D tensor" );
    //        }

    //        // NCHW: width varies fastest, then height, channels, batch
    //        strides_[ 3 ] = 1;                     // W stride
    //        strides_[ 2 ] = shape_[ 3 ];             // H stride
    //        strides_[ 1 ] = shape_[ 3 ] * shape_[ 2 ]; // C stride
    //        strides_[ 0 ] = shape_[ 3 ] * shape_[ 2 ] * shape_[ 1 ]; // N stride
    //    }

    //    /**
    //     * @brief Calculate the total number of elements
    //     */
    //    void computeSize() {
    //        size_ = 1;
    //        for ( auto dim : shape_ ) {
    //            size_ *= dim;
    //        }
    //    }

    //    /**
    //     * @brief Compute contiguous strides for the given shape and format
    //     *
    //     * @param shape Shape for which to compute strides
    //     * @param format Memory format to use
    //     * @return Vector of computed strides
    //     */
    //    static std::vector<size_t> computeContiguousStrides(
    //        const std::vector<size_t>& shape,
    //        MemoryFormat format ) {

    //        std::vector<size_t> strides( shape.size() );

    //        switch ( format ) {
    //            case MemoryFormat::RowMajor:
    //            {
    //                strides[ strides.size() - 1 ] = 1;
    //                for ( int i = strides.size() - 2; i >= 0; --i ) {
    //                    strides[ i ] = strides[ i + 1 ] * shape[ i + 1 ];
    //                }
    //                break;
    //            }

    //            case MemoryFormat::ColumnMajor:
    //            {
    //                strides[ 0 ] = 1;
    //                for ( size_t i = 1; i < strides.size(); ++i ) {
    //                    strides[ i ] = strides[ i - 1 ] * shape[ i - 1 ];
    //                }
    //                break;
    //            }

    //            case MemoryFormat::ChannelsLast:
    //            {
    //                if ( shape.size() != 4 ) {
    //                    throw std::invalid_argument( "ChannelsLast format requires 4D tensor" );
    //                }
    //                strides[ 3 ] = 1;
    //                strides[ 2 ] = shape[ 3 ];
    //                strides[ 1 ] = shape[ 3 ] * shape[ 2 ];
    //                strides[ 0 ] = shape[ 3 ] * shape[ 2 ] * shape[ 1 ];
    //                break;
    //            }

    //            case MemoryFormat::ChannelsFirst:
    //            {
    //                if ( shape.size() != 4 ) {
    //                    throw std::invalid_argument( "ChannelsFirst format requires 4D tensor" );
    //                }
    //                strides[ 3 ] = 1;
    //                strides[ 2 ] = shape[ 3 ];
    //                strides[ 1 ] = shape[ 3 ] * shape[ 2 ];
    //                strides[ 0 ] = shape[ 3 ] * shape[ 2 ] * shape[ 1 ];
    //                break;
    //            }
    //        }

    //        return strides;
    //    }
    //};

    /**
     * @brief CUDA/cuBLASLt-specific tensor layout interface
     *
     * Provides specialized layout functionality for CUDA operations,
     * particularly cuBLASLt matrix multiplications.
     */
    //export class CudaTensorLayout : public StandardTensorLayout {
    //public:
    //    /**
    //     * @brief Construct a new CUDA-optimized tensor layout
    //     *
    //     * @param dimensions Vector specifying tensor dimensions
    //     * @param format Memory layout format
    //     * @param alignment Memory alignment in bytes (typically 16 for CUDA)
    //     */
    //    CudaTensorLayout(
    //        const std::vector<size_t>& dimensions,
    //        MemoryFormat format = MemoryFormat::RowMajor,
    //        size_t alignment = 16 )
    //        : StandardTensorLayout( dimensions, format ),
    //        alignment_( alignment ) {

    //        // Adjust strides for alignment if needed
    //        alignStrides();
    //    }

    //    /**
    //     * @brief Convert to cublasLtMatrixLayout for use with cuBLASLt
    //     *
    //     * @param dataType CUDA data type enum value
    //     * @return Newly created cuBLASLt matrix layout
    //     * @note Caller is responsible for destroying the returned layout
    //     */
    //    cublasLtMatrixLayout_t toCublasLtLayout( cudaDataType_t dataType ) const {
    //        if ( rank() < 2 ) {
    //            throw std::runtime_error( "cuBLASLt requires at least a 2D tensor" );
    //        }

    //        cublasLtMatrixLayout_t layout = nullptr;

    //        if ( rank() == 2 ) {
    //            // For 2D tensors, use rows and columns directly
    //            size_t rows = shape_[ 0 ];
    //            size_t cols = shape_[ 1 ];
    //            size_t ld = leadingDimension();

    //            // Create native layout
    //            cublasLtMatrixLayoutCreate( &layout, dataType, rows, cols, ld );
    //        }
    //        else {
    //            // For higher-rank tensors, treat as batched matrices
    //            // Flatten all but the last two dimensions into batch
    //            size_t batch = 1;
    //            for ( size_t i = 0; i < rank() - 2; ++i ) {
    //                batch *= shape_[ i ];
    //            }

    //            size_t rows = shape_[ rank() - 2 ];
    //            size_t cols = shape_[ rank() - 1 ];
    //            size_t ld = leadingDimension();

    //            // Create batched layout
    //            cublasLtMatrixLayoutCreate( &layout, dataType, rows, cols, ld );

    //            // Set batch size and stride
    //            size_t batchStride = rows * cols;
    //            cublasLtMatrixLayoutSetAttribute(
    //                layout,
    //                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
    //                &batch,
    //                sizeof( batch ) );

    //            cublasLtMatrixLayoutSetAttribute(
    //                layout,
    //                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
    //                &batchStride,
    //                sizeof( batchStride ) );
    //        }

    //        return layout;
    //    }

    //    /**
    //     * @brief Create a transposed layout specifically optimized for CUDA
    //     *
    //     * @return Unique pointer to new CUDA-optimized tensor layout
    //     */
    //    std::unique_ptr<TensorLayout> transpose() const override {
    //        if ( rank() < 2 ) {
    //            return std::make_unique<CudaTensorLayout>( shape_, format_, alignment_ );
    //        }

    //        std::vector<size_t> transposedShape( shape_ );

    //        if ( rank() == 2 ) {
    //            // Swap rows and columns
    //            std::swap( transposedShape[ 0 ], transposedShape[ 1 ] );
    //        }
    //        else {
    //            // For higher-rank tensors, transpose the last two dimensions
    //            std::swap(
    //                transposedShape[ rank() - 1 ],
    //                transposedShape[ rank() - 2 ] );
    //        }

    //        // Invert the memory format
    //        MemoryFormat newFormat = (format_ == MemoryFormat::RowMajor) ?
    //            MemoryFormat::ColumnMajor : MemoryFormat::RowMajor;

    //        return std::make_unique<CudaTensorLayout>( transposedShape, newFormat, alignment_ );
    //    }

    //    /**
    //     * @brief Get the memory alignment
    //     * @return Alignment in bytes
    //     */
    //    size_t alignment() const {
    //        return alignment_;
    //    }

    //    /**
    //     * @brief Create string representation including CUDA-specific info
    //     * @return String describing the layout
    //     */
    //    std::string toString() const override {
    //        std::string result = StandardTensorLayout::toString();

    //        // Remove the trailing parenthesis to add more info
    //        result.pop_back();

    //        // Add CUDA-specific information
    //        result += ", alignment=" + std::to_string( alignment_ ) + ")";

    //        return result;
    //    }

    //private:
    //    size_t alignment_; ///< Memory alignment in bytes

    //    /**
    //     * @brief Adjust strides to ensure aligned memory accesses
    //     */
    //    void alignStrides() {
    //        // Ensure the fastest-changing dimension's stride is at least 1
    //        size_t fastDimIndex = (format_ == MemoryFormat::RowMajor) ?
    //            rank() - 1 : 0;

    //        if ( strides_[ fastDimIndex ] != 1 ) {
    //            return; // Already aligned or non-standard layout
    //        }

    //        // Calculate element size based on a reasonable assumption
    //        // This would need to be parameterized in a real implementation
    //        constexpr size_t assumedElemSize = 4; // Assume 4 bytes (e.g., float)

    //        // Calculate the next dimension's stride with alignment
    //        size_t nextDimIndex = (format_ == MemoryFormat::RowMajor) ?
    //            rank() - 2 : 1;

    //        if ( nextDimIndex >= rank() ) {
    //            return; // Only one dimension
    //        }

    //        // Calculate ideal aligned stride
    //        size_t elementsPerLine = shape_[ fastDimIndex ];
    //        size_t bytesPerLine = elementsPerLine * assumedElemSize;
    //        size_t alignedBytesPerLine = ((bytesPerLine + alignment_ - 1) / alignment_) * alignment_;
    //        size_t alignedElementsPerLine = alignedBytesPerLine / assumedElemSize;

    //        // Update the stride if necessary
    //        if ( alignedElementsPerLine > elementsPerLine ) {
    //            // Adjust all strides to maintain proper layout
    //            size_t strideAdjustment = alignedElementsPerLine - elementsPerLine;

    //            if ( format_ == MemoryFormat::RowMajor ) {
    //                // For row-major, need to adjust all strides except the last one
    //                for ( size_t i = 0; i < rank() - 1; ++i ) {
    //                    strides_[ i ] += strideAdjustment;
    //                }
    //            }
    //            else if ( format_ == MemoryFormat::ColumnMajor ) {
    //                // For column-major, adjust all strides except the first one
    //                for ( size_t i = 1; i < rank(); ++i ) {
    //                    strides_[ i ] += strideAdjustment * strides_[ 0 ];
    //                }
    //            }

    //            // Recalculate size to account for padding
    //            size_ = strides_[ 0 ] * shape_[ 0 ];
    //        }
    //    }
    //};
}