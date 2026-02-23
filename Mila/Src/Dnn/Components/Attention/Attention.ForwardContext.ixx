export module Dnn.Components.Attention:ForwardContext;

namespace Mila::Dnn
{
    export struct AttentionForwardContext
    {
        enum class Mode
        {
            Standard, Prefill, Decode
        } 
        
        mode = Mode::Standard;
        int position = 0;
    };
}