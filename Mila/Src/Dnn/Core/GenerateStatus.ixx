module;
#include <optional>
#include <string_view>
#include <cstdint>

export module Dnn.GenerateStatus;

namespace Mila::Dnn
{
    export enum class [[nodiscard]] GenerateStatus : int32_t
    {
        Success = 0,
        MaxNewTokensReached,
        ContextOverflow,
        ClientCancelled
    };

    export inline std::string_view to_string( GenerateStatus status )
    {
        switch ( status )
        {
            case GenerateStatus::Success:
                return "stop";
            case GenerateStatus::MaxNewTokensReached:
                return "length";
            case GenerateStatus::ContextOverflow:
                return "context_limit";
            case GenerateStatus::ClientCancelled:
                return "cancelled";
        }

        return "unknown";
    }
}
