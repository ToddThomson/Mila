module;
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <vector>

export module Visualization.FrameBuffer;

namespace Mila::Dnn::Visualization
{
    export struct Rect
    {
        int x, y, w, h;
    };
    
    export struct RGB
    {
        uint8_t r, g, b;
    };

    export class Framebuffer
    {
    private:
        int width, height;
        std::vector<uint8_t> pixels;
    
    public:
    
        Framebuffer( int w = 3840, int h = 2160 ) : width( w ), height( h ), pixels( w* h * 3, 0 )
        {
        }
        
        inline uint8_t* pixelPtr( int x, int y )
        {
            return &pixels[(y * width + x) * 3];
        }
        
        void clear( uint8_t r = 0, uint8_t g = 0, uint8_t b = 0 )
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    uint8_t* p = pixelPtr( x, y );
                    p[0] = r; p[1] = g; p[2] = b;
                }
            }
        }
    };
}