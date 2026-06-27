module;
#include <optional>
#include <string_view>

export module Dnn.GenerateStatus;

namespace Mila::Dnn
{
    enum class [[nodiscard]] GenerateStatus : int32_t
    {
        Success = 0,
        MaxNewTokensReached,
        ContextOverflow,
        ClientCancelled
    };

    inline std::string_view to_string( GenerateStatus status )
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
