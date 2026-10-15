/**
 * @file Chat.KvCache.ixx
 * @brief How a Chat session holds its context, as the library names it, and Chat's keyword for each.
 */

export module Chat.KvCache;

import std;
import Mila;

namespace Mila::ChatApp
{
    /// The library's KV-cache setting, so the one Chat asks for is the one the load builds.
    export using KvCache = Mila::Dnn::KvCacheCompression;

    /**
     * @brief Parse a KV-cache keyword ("bf16"/"fp8"). std::nullopt for anything else.
     */
    export constexpr std::optional<KvCache> parseKvCache( std::string_view s )
    {
        if ( s == "bf16" )  return KvCache::None;
        if ( s == "fp8" )   return KvCache::FP8;

        return std::nullopt;
    }

    /**
     * @brief Display name for a KvCache. The spelling parseKvCache accepts.
     */
    export constexpr std::string_view kvCacheName( KvCache kv_cache )
    {
        switch ( kv_cache )
        {
            case KvCache::FP8: return "fp8";
            case KvCache::None: return "bf16";
        }

        return "bf16";
    }
}
