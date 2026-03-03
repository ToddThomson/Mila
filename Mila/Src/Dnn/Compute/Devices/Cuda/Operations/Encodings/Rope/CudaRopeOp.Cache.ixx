/**
 * @file CudaRopeOp.Cache.ixx
 * @brief Process-wide shared cos/sin cache registry for CudaRopeOp.
 *
 * Provides RopeCacheRegistry, an implementation detail of Compute.CudaRopeOp.
 * Not exported to consumers of the module.
 */

module;
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <bit>
#include <cstdint>

export module Compute.CudaRopeOp:Cache;

import Dnn.TensorDataType;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda::Rope
{
    /**
     * @brief Process-wide shared cache for RoPE cos/sin frequency tables.
     *
     * The cos/sin tables are a pure function of (device_id, max_seq_len, head_dim,
     * base, precision). In a typical transformer every attention layer constructs a
     * CudaRopeOp with identical parameters; this registry ensures the tables are
     * allocated and filled exactly once per unique configuration and freed when the
     * last referencing op is destroyed.
     *
     * Thread safety: acquire() and release() are individually serialized by an
     * internal mutex. build_cache() is called by the first acquirer outside the
     * lock; subsequent acquirers receive is_new == false and skip the fill.
     */
    class RopeCacheRegistry
    {
    public:

        struct CacheKey
        {
            int                    device_id;
            std::size_t            max_seq_len;
            std::size_t            head_dim;
            float                  base;
            Dnn::TensorDataType    precision;

            bool operator==( const CacheKey& ) const = default;
        };

        struct CacheKeyHash
        {
            std::size_t operator()( const CacheKey& k ) const noexcept
            {
                auto mix = []( std::size_t seed, std::size_t v ) noexcept
                {
                    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
                };

                std::size_t seed = std::hash<int>{}( k.device_id );
                seed = mix( seed, std::hash<std::size_t>{}( k.max_seq_len ) );
                seed = mix( seed, std::hash<std::size_t>{}( k.head_dim ) );
                seed = mix( seed, std::hash<uint32_t>{}( std::bit_cast<uint32_t>( k.base ) ) );
                seed = mix( seed, std::hash<int>{}( static_cast<int>( k.precision ) ) );
                return seed;
            }
        };

        struct AcquireResult
        {
            void* cos_ptr;
            void* sin_ptr;
            bool  is_new;   ///< true when build_cache() must be called by the first acquirer
        };

        static RopeCacheRegistry& instance() noexcept
        {
            static RopeCacheRegistry registry;
            return registry;
        }

        /**
         * @brief Acquire a shared reference to the cos/sin cache for the given key.
         *
         * On first acquisition for a key, allocates device memory and returns
         * is_new == true so the caller fills the tables via build_cache(). Subsequent
         * acquisitions increment the reference count and return is_new == false.
         *
         * @param key         Uniquely identifies the cache configuration.
         * @param cache_bytes Byte size for one of the cos or sin arrays.
         * @returns AcquireResult with device pointers and is_new flag.
         * @throws CudaError if device memory allocation fails.
         */
        AcquireResult acquire( const CacheKey& key, std::size_t cache_bytes )
        {
            std::lock_guard lock( mutex_ );

            auto it = entries_.find( key );

            if ( it != entries_.end() )
            {
                ++it->second.ref_count;
                return { it->second.cos_ptr, it->second.sin_ptr, false };
            }

            void* cos_ptr = nullptr;
            void* sin_ptr = nullptr;

            cudaCheckStatus( cudaMalloc( &cos_ptr, cache_bytes ) );

            try
            {
                cudaCheckStatus( cudaMalloc( &sin_ptr, cache_bytes ) );
            }
            catch ( ... )
            {
                cudaFree( cos_ptr );
                throw;
            }

            entries_.emplace( key, CacheEntry{ cos_ptr, sin_ptr, 1 } );
            return { cos_ptr, sin_ptr, true };
        }

        /**
         * @brief Release a reference to the shared cache.
         *
         * Decrements the reference count. Frees device memory when it reaches zero.
         * Safe to call from destructors — cudaFree errors are silently ignored as
         * they are not actionable during cleanup.
         */
        void release( const CacheKey& key ) noexcept
        {
            std::lock_guard lock( mutex_ );

            auto it = entries_.find( key );

            if ( it == entries_.end() )
                return;

            if ( --it->second.ref_count == 0 )
            {
                cudaFree( it->second.cos_ptr );
                cudaFree( it->second.sin_ptr );
                entries_.erase( it );
            }
        }

    private:

        struct CacheEntry
        {
            void*       cos_ptr{ nullptr };
            void*       sin_ptr{ nullptr };
            std::size_t ref_count{ 0 };
        };

        std::mutex mutex_;
        std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> entries_;

        RopeCacheRegistry() = default;
        RopeCacheRegistry( const RopeCacheRegistry& ) = delete;
        RopeCacheRegistry& operator=( const RopeCacheRegistry& ) = delete;
    };
}