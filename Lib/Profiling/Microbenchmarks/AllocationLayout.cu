/**
 * @file AllocationLayout.cu
 * @brief Whether packing tensors into larger device allocations costs anything on this driver: the same tensors as one
 * allocation each, packed into slabs of several sizes, as one block, and in one reserved address range mapped a chunk
 * at a time -- allocated near the card's limit, read with no pressure, and read while a second process forces the
 * driver to evict.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    constexpr std::size_t kMiB = std::size_t{ 1 } << 20;
    constexpr std::size_t kAlignment = 256;
    constexpr std::size_t kDriverPackedLimit = kMiB;
    constexpr int kPasses = 5;

    std::size_t alignUp( std::size_t bytes )
    {
        return ( bytes + kAlignment - 1 ) / kAlignment * kAlignment;
    }

    __global__ void readKernel( const uint4* __restrict__ data, std::size_t count, unsigned* sink )
    {
        unsigned accumulator = 0;

        for ( std::size_t i = blockIdx.x * static_cast<std::size_t>( blockDim.x ) + threadIdx.x; i < count;
            i += static_cast<std::size_t>( gridDim.x ) * blockDim.x )
        {
            const uint4 value = data[ i ];
            accumulator ^= value.x ^ value.y ^ value.z ^ value.w;
        }

        // Never true for zeroed memory; it keeps the loads from being eliminated.
        if ( accumulator == 0x9e3779b9u )
            sink[ 0 ] = accumulator;
    }

    struct Placement
    {
        std::vector<void*> allocations;
        std::vector<const uint4*> regions;
        std::vector<std::size_t> region_bytes;
        std::size_t allocated_bytes{ 0 };

        CUdeviceptr range{ 0 };
        std::size_t reserved{ 0 };
        std::size_t mapped{ 0 };
        std::vector<CUmemGenericAllocationHandle> handles;
    };

    void release( Placement& placement )
    {
        for ( void* pointer : placement.allocations )
            cudaFree( pointer );

        if ( placement.range != 0 )
        {
            cuMemUnmap( placement.range, placement.mapped );

            for ( CUmemGenericAllocationHandle handle : placement.handles )
                cuMemRelease( handle );

            cuMemAddressFree( placement.range, placement.reserved );
        }

        placement = Placement{};
    }

    bool allocate( Placement& placement, std::size_t bytes, void*& pointer )
    {
        if ( cudaMalloc( &pointer, bytes ) != cudaSuccess )
        {
            cudaGetLastError();
            return false;
        }

        placement.allocations.push_back( pointer );
        placement.allocated_bytes += bytes;

        return cudaMemset( pointer, 0, bytes ) == cudaSuccess;
    }

    struct Slab
    {
        unsigned char* base;
        std::size_t used;
        std::size_t size;
    };

    // slab_bytes == 0: one allocation per tensor. SIZE_MAX: one block. Otherwise each tensor above the driver's packed
    // limit goes to the first slab with room for it, a new slab when none has, and its own allocation when it is
    // larger than a slab -- the manager's policy, so the arms measure what it would do. Tensors at or below the
    // limit are left to the driver in every arm.
    bool place( const std::vector<std::size_t>& tensors, std::size_t slab_bytes, Placement& placement )
    {
        std::vector<Slab> slabs;

        if ( slab_bytes == SIZE_MAX )
        {
            std::size_t total = 0;

            for ( std::size_t bytes : tensors )
            {
                if ( bytes > kDriverPackedLimit )
                    total += alignUp( bytes );
            }

            void* pointer = nullptr;

            if ( !allocate( placement, total, pointer ) )
                return false;

            slabs.push_back( Slab{ static_cast<unsigned char*>( pointer ), 0, total } );
        }

        for ( std::size_t bytes : tensors )
        {
            const std::size_t aligned = alignUp( bytes );
            void* pointer = nullptr;

            if ( slab_bytes == 0 || bytes <= kDriverPackedLimit || ( slab_bytes != SIZE_MAX && aligned > slab_bytes ) )
            {
                if ( !allocate( placement, bytes, pointer ) )
                    return false;
            }
            else
            {
                auto slab = std::find_if( slabs.begin(), slabs.end(),
                    [aligned]( const Slab& s ) { return s.used + aligned <= s.size; } );

                if ( slab == slabs.end() )
                {
                    void* fresh = nullptr;

                    if ( !allocate( placement, slab_bytes, fresh ) )
                        return false;

                    slabs.push_back( Slab{ static_cast<unsigned char*>( fresh ), 0, slab_bytes } );
                    slab = slabs.end() - 1;
                }

                pointer = slab->base + slab->used;
                slab->used += aligned;
            }

            placement.regions.push_back( static_cast<const uint4*>( pointer ) );
            placement.region_bytes.push_back( bytes );
        }

        return true;
    }

    // One reserved address range; tensors above the driver's packed limit are placed end to end in it, and physical
    // memory is created and mapped `chunk_bytes` at a time as the end advances. Each chunk is its own physical
    // allocation, so the driver can still evict piece by piece.
    bool placeMapped( const std::vector<std::size_t>& tensors, std::size_t chunk_bytes, Placement& placement )
    {
        CUmemAllocationProp properties{};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = 0;

        std::size_t granularity = 0;

        if ( cuMemGetAllocationGranularity( &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM ) != CUDA_SUCCESS )
            return false;

        chunk_bytes = ( chunk_bytes + granularity - 1 ) / granularity * granularity;

        std::size_t total = 0;

        for ( std::size_t bytes : tensors )
        {
            if ( bytes > kDriverPackedLimit )
                total += alignUp( bytes );
        }

        placement.reserved = ( total / chunk_bytes + 2 ) * chunk_bytes;

        if ( cuMemAddressReserve( &placement.range, placement.reserved, 0, 0, 0 ) != CUDA_SUCCESS )
        {
            placement.range = 0;
            return false;
        }

        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        std::size_t used = 0;

        for ( std::size_t bytes : tensors )
        {
            void* pointer = nullptr;

            if ( bytes <= kDriverPackedLimit )
            {
                if ( !allocate( placement, bytes, pointer ) )
                    return false;
            }
            else
            {
                const std::size_t aligned = alignUp( bytes );

                while ( placement.mapped < used + aligned )
                {
                    CUmemGenericAllocationHandle handle{};

                    if ( cuMemCreate( &handle, chunk_bytes, &properties, 0 ) != CUDA_SUCCESS )
                        return false;

                    placement.handles.push_back( handle );

                    if ( cuMemMap( placement.range + placement.mapped, chunk_bytes, 0, handle, 0 ) != CUDA_SUCCESS
                        || cuMemSetAccess( placement.range + placement.mapped, chunk_bytes, &access, 1 ) != CUDA_SUCCESS )
                        return false;

                    placement.mapped += chunk_bytes;
                    placement.allocated_bytes += chunk_bytes;
                }

                pointer = reinterpret_cast<void*>( placement.range + used );
                used += aligned;
            }

            placement.regions.push_back( static_cast<const uint4*>( pointer ) );
            placement.region_bytes.push_back( bytes );
        }

        return cudaMemset( reinterpret_cast<void*>( placement.range ), 0, placement.mapped ) == cudaSuccess;
    }

    double readPass( const Placement& placement, unsigned* sink )
    {
        cudaEvent_t start, stop;
        cudaEventCreate( &start );
        cudaEventCreate( &stop );
        cudaEventRecord( start );

        std::size_t bytes_read = 0;

        for ( std::size_t i = 0; i < placement.regions.size(); ++i )
        {
            const std::size_t count = placement.region_bytes[ i ] / sizeof( uint4 );

            if ( count == 0 )
                continue;

            const unsigned blocks = static_cast<unsigned>( std::min<std::size_t>( ( count + 255 ) / 256, 1024 ) );
            readKernel<<<blocks, 256>>>( placement.regions[ i ], count, sink );
            bytes_read += count * sizeof( uint4 );
        }

        cudaEventRecord( stop );
        cudaEventSynchronize( stop );

        float milliseconds = 0.0f;
        cudaEventElapsedTime( &milliseconds, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        return static_cast<double>( bytes_read ) / ( milliseconds * 1e6 );
    }

    double medianPass( const Placement& placement, unsigned* sink )
    {
        std::vector<double> rates;

        for ( int pass = 0; pass < kPasses; ++pass )
            rates.push_back( readPass( placement, sink ) );

        std::sort( rates.begin(), rates.end() );

        return rates[ kPasses / 2 ];
    }

    std::size_t freeBytes()
    {
        std::size_t free = 0;
        std::size_t total = 0;
        cudaMemGetInfo( &free, &total );

        return free;
    }

    void writeMarker( const std::string& path, const std::string& text )
    {
        std::ofstream( path ) << text << "\n";
    }

    bool waitForMarker( const std::string& path, int timeout_seconds )
    {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( timeout_seconds );

        while ( std::chrono::steady_clock::now() < end )
        {
            if ( std::ifstream( path ).good() )
                return true;

            std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
        }

        return false;
    }

    // The second process: holds `megabytes` and keeps writing it, so the driver must keep it resident. It marks
    // `<marker>.ready` once the memory is allocated and written, and `<marker>.done` once it is freed, so the
    // measuring process times its passes against the pressure rather than against a guess at its start-up.
    int runPressure( std::size_t megabytes, int seconds, const std::string& marker )
    {
        void* pointer = nullptr;

        if ( cudaMalloc( &pointer, megabytes * kMiB ) != cudaSuccess )
        {
            writeMarker( marker + ".ready", "failed" );
            writeMarker( marker + ".done", "failed" );
            return 1;
        }

        cudaMemset( pointer, 0, megabytes * kMiB );
        cudaDeviceSynchronize();
        writeMarker( marker + ".ready", "held" );

        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds( seconds );
        int writes = 0;

        while ( std::chrono::steady_clock::now() < end )
        {
            cudaMemset( pointer, writes & 0xff, megabytes * kMiB );
            cudaDeviceSynchronize();
            ++writes;
        }

        cudaFree( pointer );
        cudaDeviceSynchronize();
        writeMarker( marker + ".done", std::to_string( writes ) + " writes" );

        return 0;
    }
}

int main( int argc, char** argv )
{
    if ( argc >= 5 && std::string( argv[ 1 ] ) == "pressure" )
        return runPressure( std::strtoull( argv[ 2 ], nullptr, 10 ), std::atoi( argv[ 3 ] ), argv[ 4 ] );

    if ( argc < 2 )
    {
        std::printf( "AllocationLayout <tensor list: 'layer bytes' per line> [pressure MiB] [pressure seconds]\n" );
        return 2;
    }

    std::vector<std::size_t> tensors;
    std::ifstream list( argv[ 1 ] );
    long long layer = 0;
    std::size_t bytes = 0;

    while ( list >> layer >> bytes )
        tensors.push_back( bytes );

    const std::size_t pressure_megabytes = argc >= 3 ? std::strtoull( argv[ 2 ], nullptr, 10 ) : 0;
    const int pressure_seconds = argc >= 4 ? std::atoi( argv[ 3 ] ) : 20;

    std::size_t tensor_bytes = 0;

    for ( std::size_t b : tensors )
        tensor_bytes += b;

    unsigned* sink = nullptr;
    cudaMalloc( &sink, sizeof( unsigned ) );

    std::printf( "%zu tensors, %.0f MiB; free %.0f MiB; pressure %zu MiB\n", tensors.size(),
        tensor_bytes / static_cast<double>( kMiB ), freeBytes() / static_cast<double>( kMiB ), pressure_megabytes );
    std::printf( "%-12s %8s %12s %12s %9s %12s %12s %11s\n", "layout", "allocs", "requested", "consumed", "place ms",
        "quiet", "pressured", "recovered" );

    struct Arm
    {
        std::string name;
        bool mapped;
        std::size_t bytes;
    };

    const Arm arms[] = {
        { "per-tensor", false, 0 },
        { "slab-64", false, 64 * kMiB },
        { "slab-256", false, 256 * kMiB },
        { "slab-1024", false, 1024 * kMiB },
        { "one-block", false, SIZE_MAX },
        { "mapped-2", true, 2 * kMiB },
        { "mapped-64", true, 64 * kMiB },
    };

    // An arm named on the command line runs alone, so that arms measured under pressure do not inherit each other's
    // evictions; run each in its own process.
    const std::string only = argc >= 5 ? argv[ 4 ] : "";

    for ( const Arm& arm : arms )
    {
        if ( !only.empty() && arm.name != only )
            continue;

        const std::string& name = arm.name;

        Placement placement;
        const std::size_t free_before = freeBytes();
        const auto placed_at = std::chrono::steady_clock::now();
        const bool placed = ( arm.mapped ? placeMapped( tensors, arm.bytes, placement ) : place( tensors, arm.bytes, placement ) )
            && cudaDeviceSynchronize() == cudaSuccess;
        const double consumed = ( static_cast<double>( free_before ) - static_cast<double>( freeBytes() ) ) / kMiB;
        const double place_ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - placed_at ).count();

        if ( !placed )
        {
            std::printf( "%-12s allocation FAILED after %zu allocations (%.0f MiB)\n", name.c_str(),
                placement.allocations.size(), placement.allocated_bytes / static_cast<double>( kMiB ) );
            release( placement );
            continue;
        }

        readPass( placement, sink );
        const double quiet = medianPass( placement, sink );
        double pressured = 0.0;
        double recovered = 0.0;
        double free_under_pressure = 0.0;

        if ( pressure_megabytes > 0 )
        {
            const std::string marker = std::string( argv[ 1 ] ) + "." + name + ".pressure";
            std::remove( ( marker + ".ready" ).c_str() );
            std::remove( ( marker + ".done" ).c_str() );

            const std::string command = "start \"\" /b \"" + std::string( argv[ 0 ] ) + "\" pressure "
                + std::to_string( pressure_megabytes ) + " " + std::to_string( pressure_seconds ) + " \"" + marker + "\"";
            std::system( command.c_str() );

            if ( !waitForMarker( marker + ".ready", 120 ) )
            {
                std::printf( "%-12s the pressure process never reported holding its memory\n", name.c_str() );
                release( placement );
                continue;
            }

            std::string held;
            std::getline( std::ifstream( marker + ".ready" ), held );

            if ( held != "held" )
            {
                std::printf( "%-12s the pressure process could not allocate %zu MiB\n", name.c_str(), pressure_megabytes );
                release( placement );
                continue;
            }

            free_under_pressure = freeBytes() / static_cast<double>( kMiB );
            pressured = medianPass( placement, sink );

            if ( !waitForMarker( marker + ".done", pressure_seconds + 120 ) )
            {
                std::printf( "%-12s the pressure process never reported freeing its memory\n", name.c_str() );
            }

            recovered = medianPass( placement, sink );
        }

        std::printf( "%-12s %8zu %8.0f MiB %8.0f MiB %9.0f %7.1f GB/s %7.1f GB/s %6.1f GB/s  free under pressure %.0f MiB\n",
            name.c_str(), placement.allocations.size(), placement.allocated_bytes / static_cast<double>( kMiB ), consumed,
            place_ms, quiet, pressured, recovered, free_under_pressure );

        release( placement );
    }

    cudaFree( sink );

    return 0;
}
