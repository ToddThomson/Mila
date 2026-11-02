module;
#include <nvtx3/nvtx3.hpp>

export module Profiling.NvtxRange;

import std;

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