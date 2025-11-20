module;
#include <cstdint>
#include <algorithm>
#include <cmath>

export module Visualization.HeatmapRenderer;

import Visualization.FrameBuffer;
import Visualization.ColorLUT;

import Dnn.TensorTypes;
import Dnn.ITensor;

namespace Mila::Dnn::Visualization
{
    static void renderHeatmapAvg( const ITensor& t, const Rect& rect, Framebuffer& fb, const ColorLUT& lut,
        bool diverging = true, float clip_abs = 0.0f )
    {
        //// Determine scale factors
        //int in_h = t.rows, in_w = t.cols;

        //for (int py = 0; py < rect.h; ++py)
        //{
        //    // source v range
        //    int sy0 = (py * in_h) / rect.h;
        //    int sy1 = ((py + 1) * in_h) / rect.h;
        //    if (sy1 <= sy0) sy1 = sy0 + 1;
        //    for (int px = 0; px < rect.w; ++px)
        //    {
        //        int sx0 = (px * in_w) / rect.w;
        //        int sx1 = ((px + 1) * in_w) / rect.w;
        //        if (sx1 <= sx0) sx1 = sx0 + 1;
        //        double acc = 0.0;
        //        int count = 0;
        //        for (int sy = sy0; sy < sy1; ++sy) for (int sx = sx0; sx < sx1; ++sx)
        //        {
        //            acc += t.data[sy * in_w + sx];
        //            ++count;
        //        }
        //        float avg = float( acc / max( 1, count ) );
        //        // if requested, clip using absolute value (useful for diverging)
        //        float value = avg;
        //        if (clip_abs > 0.0f)
        //        {
        //            if (value > clip_abs) value = clip_abs;
        //            if (value < -clip_abs) value = -clip_abs;
        //        }
        //        // map to LUT index
        //        int idx;
        //        if (diverging)
        //        {
        //            // value expected in [-clip_abs, +clip_abs] or unknown; we'll compress by tanh if needed
        //            float v = value;
        //            if (clip_abs <= 0.0f)
        //            {
        //                // autoscale: use tanh to compress dynamic range
        //                v = std::tanh( value );
        //                idx = int( (v * 0.5f + 0.5f) * (int( lut.diverging.size() ) - 1) );
        //            }
        //            else
        //            {
        //                float n = (v / clip_abs + 1.0f) * 0.5f;
        //                n = std::min( 1.0f, std::max( 0.0f, n ) );
        //                idx = int( n * (int( lut.diverging.size() ) - 1) );
        //            }
        //            RGB c = lut.diverging[idx];
        //            uint8_t* p = fb.pixelPtr( rect.x + px, rect.y + py );
        //            p[0] = c.r; p[1] = c.g; p[2] = c.b;
        //        }
        //        else
        //        {
        //            // sequential map expecting [0..1]
        //            float v = value;
        //            if (v < 0.0f) v = 0.0f;
        //            // if clip_abs>0 use clip_abs as scale
        //            if (clip_abs > 0.0f) v = std::min( 1.0f, v / clip_abs );
        //            // compress
        //            v = sqrt( v );
        //            idx = int( v * (int( lut.sequential.size() ) - 1) );
        //            RGB c = lut.sequential[idx];
        //            uint8_t* p = fb.pixelPtr( rect.x + px, rect.y + py );
        //            p[0] = c.r; p[1] = c.g; p[2] = c.b;
        //        }
        //    }
        //}
    };
}