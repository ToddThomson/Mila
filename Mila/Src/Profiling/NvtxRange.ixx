module;
#include <nvtx3/nvtx3.hpp>
#include <string_view>

export module Profiling.NvtxRange;

namespace Mila::Profiling
{
    export class NvtxRange
    {
    public:

        explicit NvtxRange( std::string_view name )
        {
            nvtxRangePushA( name.data() );
        }

        ~NvtxRange()
        {
            nvtxRangePop();
        }

        NvtxRange( const NvtxRange& ) = delete;
        NvtxRange& operator=( const NvtxRange& ) = delete;
        NvtxRange( NvtxRange&& ) = delete;
        NvtxRange& operator=( NvtxRange&& ) = delete;
    };
}