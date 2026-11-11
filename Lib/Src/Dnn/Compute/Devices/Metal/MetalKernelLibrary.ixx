/**
 * @file MetalKernelLibrary.ixx
 * @brief Mila's Metal kernels: compiled when Mila is built, and carried in the library.
 */

module;

// Only the names: each module that includes metal-cpp costs every importer of Mila (MetalBackend.md section 5).
namespace MTL
{
    class ComputePipelineState;
    class Device;
    class Library;
}

export module Compute.MetalKernelLibrary;

import std;
import Compute.MetalKernelLimits;

namespace Mila::Dnn::Compute
{
    /// A kernel ready to dispatch. Held by the library that handed it out, and valid for that library's lifetime.
    export using MetalPipeline = MTL::ComputePipelineState*;

    /**
     * @brief The kernels of one device.
     *
     * The kernels are compiled by the build into one Metal library, embedded in Mila and loaded when a kernel is
     * first asked for; a pipeline is created once per kernel (MetalBackend.md section 4.3). An operation asks for
     * its pipelines when it is constructed, so that a kernel Mila does not have is refused when a model loads and
     * no dispatch looks one up by name. Safe to call from several threads. The bodies are in
     * Implementation/MetalKernelLibrary.cpp, with the compiled kernels.
     */
    export class MetalKernelLibrary
    {
    public:

        explicit MetalKernelLibrary( MTL::Device* device );

        ~MetalKernelLibrary();

        MetalKernelLibrary( const MetalKernelLibrary& ) = delete;
        MetalKernelLibrary& operator=( const MetalKernelLibrary& ) = delete;

        /**
         * @brief The pipeline of a kernel.
         *
         * @throws std::runtime_error When Metal refuses the embedded library, the kernel does not exist, or its
         *         pipeline cannot be created. Each says why.
         */
        [[nodiscard]] MetalPipeline pipeline( const std::string& kernel_name );

        /// The name of every kernel Mila ships.
        [[nodiscard]] std::vector<std::string> kernelNames();

        /// The kernel a pipeline of this library runs, for a profile's labels; empty for one it did not make.
        [[nodiscard]] std::string nameOf( const MTL::ComputePipelineState* state );

        [[nodiscard]] MetalKernelLimits limits( const std::string& kernel_name );

    private:

        /// The kernels' library, loaded from the embedded bytes on the first call. Called with the lock held.
        MTL::Library* loaded();

        MTL::Device* device_;
        MTL::Library* library_{ nullptr };
        std::mutex mutex_;
        std::unordered_map<std::string, MTL::ComputePipelineState*> pipelines_;
    };
}
