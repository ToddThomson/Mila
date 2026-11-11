/**
 * @file MetalKernelLibrary.cpp
 * @brief The bodies of MetalKernelLibrary's members and the compiled kernels; the interface is MetalKernelLibrary.ixx.
 */

module;

#include <dispatch/dispatch.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

module Compute.MetalKernelLibrary;

import std;
import Compute.MetalKernelLimits;

namespace Mila::Dnn::Compute
{
    namespace
    {
        // Every kernel, compiled by the build from the files Lib/CMakeLists.txt lists (MetalBackend.md section 4.3).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
        const unsigned char kKernelLibrary[] = {
#embed "MilaKernels.metallib"
        };
#pragma clang diagnostic pop

        // Seen once: an #embed the compiler could not find gave an empty array and no error.
        static_assert( sizeof( kKernelLibrary ) > 4, "the build did not embed Mila's Metal kernels" );

        // Objects Metal hands back autoreleased live until this goes out of scope.
        class LocalPool
        {
        public:

            LocalPool()
                : pool_( NS::AutoreleasePool::alloc()->init() )
            {
            }

            ~LocalPool()
            {
                pool_->release();
            }

            LocalPool( const LocalPool& ) = delete;
            LocalPool& operator=( const LocalPool& ) = delete;

        private:

            NS::AutoreleasePool* pool_;
        };

        std::string describe( const NS::Error* error )
        {
            return error ? error->localizedDescription()->utf8String() : "no reason given";
        }
    }

    MetalKernelLibrary::MetalKernelLibrary( MTL::Device* device )
        : device_( device )
    {
    }

    MetalKernelLibrary::~MetalKernelLibrary()
    {
        for ( auto& [name, pipeline] : pipelines_ )
        {
            pipeline->release();
        }

        if ( library_ )
        {
            library_->release();
        }
    }

    MetalPipeline MetalKernelLibrary::pipeline( const std::string& kernel_name )
    {
        std::lock_guard lock( mutex_ );

        const auto found = pipelines_.find( kernel_name );

        if ( found != pipelines_.end() )
        {
            return found->second;
        }

        MTL::Library* const library = loaded();
        LocalPool pool;

        MTL::Function* const function = library->newFunction( NS::String::string( kernel_name.c_str(), NS::UTF8StringEncoding ) );

        if ( !function )
        {
            throw std::runtime_error( std::format( "Mila has no Metal kernel named '{}'", kernel_name ) );
        }

        NS::Error* error = nullptr;
        MTL::ComputePipelineState* const created = device_->newComputePipelineState( function, &error );
        function->release();

        if ( !created )
        {
            throw std::runtime_error( std::format( "The Metal kernel '{}' could not be prepared: {}", kernel_name, describe( error ) ) );
        }

        pipelines_.emplace( kernel_name, created );

        return created;
    }

    std::vector<std::string> MetalKernelLibrary::kernelNames()
    {
        std::lock_guard lock( mutex_ );

        MTL::Library* const library = loaded();
        LocalPool pool;
        const NS::Array* const names = library->functionNames();

        std::vector<std::string> result;
        result.reserve( names->count() );

        for ( NS::UInteger i = 0; i < names->count(); ++i )
        {
            result.emplace_back( names->object<NS::String>( i )->utf8String() );
        }

        std::ranges::sort( result );

        return result;
    }

    std::string MetalKernelLibrary::nameOf( const MTL::ComputePipelineState* state )
    {
        std::lock_guard lock( mutex_ );

        const auto found = std::ranges::find_if( pipelines_, [&]( const auto& entry ) { return entry.second == state; } );

        return found != pipelines_.end() ? found->first : std::string{};
    }

    MetalKernelLimits MetalKernelLibrary::limits( const std::string& kernel_name )
    {
        const MTL::ComputePipelineState* const state = pipeline( kernel_name );

        return MetalKernelLimits{ state->threadExecutionWidth(), state->maxTotalThreadsPerThreadgroup() };
    }

    MTL::Library* MetalKernelLibrary::loaded()
    {
        if ( library_ )
        {
            return library_;
        }

        LocalPool pool;
        NS::Error* error = nullptr;

        // The default destructor has dispatch copy the bytes, so the library holds nothing of this file's.
        const dispatch_data_t data = dispatch_data_create( kKernelLibrary, sizeof( kKernelLibrary ), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT );

        library_ = device_->newLibrary( data, &error );
        dispatch_release( data );

        if ( !library_ )
        {
            throw std::runtime_error( std::format( "Metal refused Mila's kernels: {}", describe( error ) ) );
        }

        return library_;
    }
}
