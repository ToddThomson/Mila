module;

export module Compute.IDecode;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    export struct IDecode
    {
        virtual ~IDecode() = default;

        virtual void decode( const ITensor& input, ITensor& output ) const = 0;
    };
}