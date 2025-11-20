/* ---------------------------
   Colormap LUTs (fast integer lookup)
   - Diverging (blue-white-red)
   - Sequential (grayscale->hot)
   --------------------------- */

module;
#include <vector>
#include <cstdint>

export module Visualization.ColorLUT;

import Visualization.FrameBuffer;

namespace Mila::Dnn::Visualization
{
    export struct ColorLUT
    {
        std::vector<RGB> diverging; // 1024
        std::vector<RGB> sequential; // 1024

        ColorLUT( int N = 1024 )
        {
            diverging.resize( N );
            sequential.resize( N );

            for (int i = 0; i < N; ++i)
            {
                float t = float( i ) / (N - 1); // 0..1
                // Diverging: blue -> white -> red
                if (t < 0.5f)
                {
                    float u = t / 0.5f;
                    // blue(0,0,180) -> white(255,255,255)
                    diverging[i].r = uint8_t( 255 * u + 0 * (1 - u) );
                    diverging[i].g = uint8_t( 255 * u + 0 * (1 - u) );
                    diverging[i].b = uint8_t( 255 * u + 180 * (1 - u) );
                }
                else
                {
                    float u = (t - 0.5f) / 0.5f;
                    // white -> red(180,0,0)
                    diverging[i].r = uint8_t( 255 * (1 - u) + 180 * u );
                    diverging[i].g = uint8_t( 255 * (1 - u) + 0 * u );
                    diverging[i].b = uint8_t( 255 * (1 - u) + 0 * u );
                }
                
                // Sequential: black -> warm yellowish -> white
                {
                    float u = t;
                    sequential[i].r = uint8_t( std::min( 255.0f, 255.0f * (u * 1.2f) ) );
                    sequential[i].g = uint8_t( std::min( 255.0f, 200.0f * (u * 1.1f) ) );
                    sequential[i].b = uint8_t( std::min( 255.0f, 150.0f * u ) );
                }
            }
        }
    };
}