module;
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <format>
#include <functional>
#include <stdexcept>

export module Compute.CublasLtPlanCache;

import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Computes optimal bucket boundaries for cuBLASLt plan caching
     *        based on CUDA device architecture.
     *
     * Bucket sizes are aligned to tensor core tile granularity:
     *   Pre-Volta  (SM<70): grain=32 (warp-aligned, no tensor cores)
     *   Volta      (SM70):  grain=8  (FP16 tensor cores, 8x8x4 MMA)
     *   Turing     (SM75):  grain=16
     *   Ampere+    (SM80+): grain=16 (TF32/BF16/FP16, 16x8x16 MMA)
     *   Hopper     (SM90+): grain=16 (FP8 wgmma)
     */
    export inline std::vector<int> computeArchitectureBuckets( int max_batch_size )
    {
        int device = 0;
        cudaGetDevice( &device );
        cudaDeviceProp prop{};
        cudaGetDeviceProperties( &prop, device );

        const int sm = prop.major * 10 + prop.minor;

        int grain = 32;
        if ( sm >= 90 ) grain = 16;  // Hopper
        else if ( sm >= 80 ) grain = 16;  // Ampere / Ada
        else if ( sm >= 75 ) grain = 16;  // Turing
        else if ( sm >= 70 ) grain = 8;   // Volta

        //Utils::Logger::info( std::format(
        //    "CublasLtPlanCache: device='{}' SM={}.{} grain={} max_batch={}",
        //    prop.name, prop.major, prop.minor, grain, max_batch_size ) );

        std::vector<int> buckets;

        // Decode: always exact
        buckets.push_back( 1 );

        // Fine-grained region: every grain up to 8*grain
        // Covers short prompts with minimal padding waste (< 2x)
        for ( int b = grain; b <= 8 * grain; b += grain )
            if ( b < max_batch_size )
                buckets.push_back( b );

        // Medium region: powers of 2 from 16*grain up to max
        for ( int b = 16 * grain; b < max_batch_size; b *= 2 )
            buckets.push_back( b );

        // Always include max for training / full-length prefill
        buckets.push_back( max_batch_size );

        // Deduplicate and sort
        std::sort( buckets.begin(), buckets.end() );
        buckets.erase( std::unique( buckets.begin(), buckets.end() ), buckets.end() );

        std::string bucket_str;
        for ( int b : buckets )
        {
            bucket_str += std::to_string( b ) + " ";
        }
        
        //Utils::Logger::info( std::format(
        //    "CublasLtPlanCache: {} buckets: [ {}]", buckets.size(), bucket_str ) );

        return buckets;
    }

    /**
     * @brief Fast O(log N) bucket lookup.
     *        Returns smallest bucket >= batch_size.
     */
    export inline int getBucket( const std::vector<int>& buckets, int batch_size )
    {
        auto it = std::lower_bound( buckets.begin(), buckets.end(), batch_size );
        if ( it == buckets.end() )
            return buckets.back();
        return *it;
    }

    /**
     * @brief Generic plan cache keyed on batch size bucket.
     *
     * TPlan must be move-constructible (e.g. CublasLtMatMulPlan<T>).
     * Plans are built eagerly at construction time for all buckets.
     *
     * Usage:
     *   CublasLtPlanCache<CublasLtMatMulPlan<float>> cache(
     *       max_batch_size,
     *       [&]( int bucket ) { return build_my_plan( bucket ); } );
     *
     *   const auto& plan = cache.get( actual_batch_size );
     */
    export template <typename TPlan>
    class CublasLtPlanCache
    {
    public:
        using PlanBuilder = std::function<TPlan( int bucket )>;

        CublasLtPlanCache() = default;

        /**
         * @brief Construct and eagerly build all plans.
         * @param max_batch_size  Upper bound (training / max seq len)
         * @param builder         Callable: int bucket -> TPlan
         */
        CublasLtPlanCache( int max_batch_size, PlanBuilder builder )
        {
            buckets_ = computeArchitectureBuckets( max_batch_size );
            for ( int bucket : buckets_ )
            {
                cache_.emplace( bucket, builder( bucket ) );
            }
        }

        /**
         * @brief Get the plan for the smallest bucket >= batch_size.
         */
        const TPlan& get( int batch_size ) const
        {
            const int bucket = getBucket( buckets_, batch_size );
            auto it = cache_.find( bucket );
            
            if ( it == cache_.end() )
            {
                throw std::runtime_error( std::format(
                    "CublasLtPlanCache: no plan for bucket {}", bucket ) );
            }

            return it->second;
        }

        bool empty() const
        {
            return cache_.empty();
        }
        size_t size() const
        {
            return cache_.size();
        }
        const std::vector<int>& buckets() const
        {
            return buckets_;
        }

    private:
        std::vector<int> buckets_;
        std::unordered_map<int, TPlan> cache_;
    };
}