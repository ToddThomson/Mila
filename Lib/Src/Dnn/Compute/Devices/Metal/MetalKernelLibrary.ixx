/**
 * @file MetalKernelLibrary.ixx
 * @brief Mila's Metal kernels: compiled from source carried in the library, on first use.
 */

module;

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

export module Compute.MetalKernelLibrary;

import std;
import Compute.MetalKernelLimits;

namespace Mila::Dnn::Compute
{
    namespace
    {
        // Each file under Kernels/, embedded at configure time by Lib/CMakeLists.txt.
        const char* const kKernelSources[] = {
#include "TensorOps.Fill.metal.inc"
        };

        std::string describe( const NS::Error* error )
        {
            return error ? error->localizedDescription()->utf8String() : "no reason given";
        }
    }

    /**
     * @brief The kernels of one device.
     *
     * The source is compiled once, when a kernel is first asked for, and a pipeline is created once per kernel
     * (MetalBackend.md section 4.3). Safe to call from several threads.
     */
    export class MetalKernelLibrary
    {
    public:

        explicit MetalKernelLibrary( MTL::Device* device )
            : device_( device )
        {
        }

        ~MetalKernelLibrary()
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

        MetalKernelLibrary( const MetalKernelLibrary& ) = delete;
        MetalKernelLibrary& operator=( const MetalKernelLibrary& ) = delete;

        /**
         * @brief The pipeline of a kernel. Held by this library, and valid for its lifetime.
         *
         * @throws std::runtime_error When the source does not compile, the kernel does not exist, or its pipeline
         *         cannot be created. Each says why.
         */
        [[nodiscard]] MTL::ComputePipelineState* pipeline( const std::string& kernel_name )
        {
            std::lock_guard lock( mutex_ );

            const auto found = pipelines_.find( kernel_name );

            if ( found != pipelines_.end() )
            {
                return found->second;
            }

            MTL::Library* const library = compiled();
            NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();

            MTL::Function* const function = library->newFunction( NS::String::string( kernel_name.c_str(), NS::UTF8StringEncoding ) );

            if ( !function )
            {
                pool->release();

                throw std::runtime_error( std::format( "Mila has no Metal kernel named '{}'", kernel_name ) );
            }

            NS::Error* error = nullptr;
            MTL::ComputePipelineState* const created = device_->newComputePipelineState( function, &error );
            function->release();

            const std::string reason = created ? std::string{} : describe( error );
            pool->release();

            if ( !created )
            {
                throw std::runtime_error( std::format( "The Metal kernel '{}' could not be prepared: {}", kernel_name, reason ) );
            }

            pipelines_.emplace( kernel_name, created );

            return created;
        }

        /// The name of every kernel Mila ships.
        [[nodiscard]] std::vector<std::string> kernelNames()
        {
            std::lock_guard lock( mutex_ );

            MTL::Library* const library = compiled();
            NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
            const NS::Array* const names = library->functionNames();

            std::vector<std::string> result;
            result.reserve( names->count() );

            for ( NS::UInteger i = 0; i < names->count(); ++i )
            {
                result.emplace_back( names->object<NS::String>( i )->utf8String() );
            }

            pool->release();
            std::ranges::sort( result );

            return result;
        }

        [[nodiscard]] MetalKernelLimits limits( const std::string& kernel_name )
        {
            const MTL::ComputePipelineState* const state = pipeline( kernel_name );

            return MetalKernelLimits{ state->threadExecutionWidth(), state->maxTotalThreadsPerThreadgroup() };
        }

    private:

        /// The compiled library, compiling it on the first call. Called with the lock held.
        MTL::Library* compiled()
        {
            if ( library_ )
            {
                return library_;
            }

            std::string source;

            for ( const char* const kernel_source : kKernelSources )
            {
                source += kernel_source;
            }

            NS::AutoreleasePool* const pool = NS::AutoreleasePool::alloc()->init();
            NS::Error* error = nullptr;

            MTL::CompileOptions* const options = MTL::CompileOptions::alloc()->init();
            options->setLanguageVersion( MTL::LanguageVersion4_1 );
            library_ = device_->newLibrary( NS::String::string( source.c_str(), NS::UTF8StringEncoding ), options, &error );
            options->release();

            const std::string reason = library_ ? std::string{} : describe( error );
            pool->release();

            if ( !library_ )
            {
                throw std::runtime_error( std::format( "Mila's Metal kernels did not compile: {}", reason ) );
            }

            return library_;
        }

        MTL::Device* device_;
        MTL::Library* library_{ nullptr };
        std::mutex mutex_;
        std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
    };
}
