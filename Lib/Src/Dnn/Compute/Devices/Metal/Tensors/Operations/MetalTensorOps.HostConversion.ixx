/**
 * @file MetalTensorOps.HostConversion.ixx
 * @brief Converting elements on the host, by the compiler's own conversions.
 */

export module Compute.MetalHostConversion;

import std;

namespace Mila::Dnn::Compute::Metal
{
    /**
     * @brief Convert `count` elements, through `float` when either side is a floating point type.
     *
     * Clang has no direct conversion between `__fp16` and `__bf16`, and every FP16 and BF16 value is exact in `float`,
     * so going through it rounds only once, on the way to the destination.
     */
    export template<typename TSource, typename TDestination>
    void convertElements( const TSource* source, TDestination* destination, std::size_t count )
    {
        constexpr bool both_integers = std::is_integral_v<TSource> && std::is_integral_v<TDestination>;

        for ( std::size_t i = 0; i < count; ++i )
        {
            if constexpr ( both_integers )
            {
                destination[i] = static_cast<TDestination>( source[i] );
            } else {
                destination[i] = static_cast<TDestination>( static_cast<float>( source[i] ) );
            }
        }
    }
}
