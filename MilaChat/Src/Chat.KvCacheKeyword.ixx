/**
 * @file Chat.KvCacheKeyword.ixx
 * @brief Chat's keyword for each way the library holds a context.
 */

export module Chat.KvCacheKeyword;

import std;
import Dnn.KvCacheCompression;

namespace Mila::ChatApp
{
    /**
     * @brief Parse a KV-cache keyword ("bf16"/"fp8"). std::nullopt for anything else.
     */
    export constexpr std::optional<Dnn::KvCacheCompression> parseKvCache( std::string_view s )
    {
        if ( s == "bf16" )  return Dnn::KvCacheCompression::None;
        if ( s == "fp8" )   return Dnn::KvCacheCompression::FP8;

        return std::nullopt;
    }

    /**
     * @brief Display name for a KV cache. The spelling parseKvCache accepts.
     */
    export constexpr std::string_view kvCacheName( Dnn::KvCacheCompression kv_cache )
    {
        switch ( kv_cache )
        {
            case Dnn::KvCacheCompression::FP8: return "fp8";
            case Dnn::KvCacheCompression::None: return "bf16";
        }

        return "bf16";
    }
}
