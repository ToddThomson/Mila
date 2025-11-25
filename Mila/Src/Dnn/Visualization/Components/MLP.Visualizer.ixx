/* ---------------------------
   MLP Visualizer
   Renders 3 heatmaps: Layer1 weights, activations, Layer2 weights
   ---------------------------
*/

module;

export module Visualization.MLP;

import Visualization.Model;
import Visualization.Context;
import Visualization.FrameBuffer;
import Visualization.ColorLUT;
import Visualization.HeatmapRenderer;

namespace Mila::Dnn::Visualization
{
    export class MLPVisualizer : ModuleVisualizer
    {
	private:
        VisualizerContext* ctx;
        const ColorLUT& lut;
        float clip_abs;
    
    public:
    
        MLPVisualizer( VisualizerContext* c, const ColorLUT& l, float clip = 0.0f ) 
            : ctx( c ), lut( l ), clip_abs( clip )
        {
        }
        
        void render( Framebuffer& fb, const Rect& region ) override
        {
            int w = region.w, h = region.h;
            int each_h = h / 3;
            Rect r1{ region.x, region.y, w, each_h };
            Rect ract{ region.x, region.y + each_h, w, each_h };
            Rect r2{ region.x, region.y + 2 * each_h, w, h - 2 * each_h };
            
            //renderHeatmapAvg( ctx->mlp_l1, r1, fb, lut, true, clip_abs );
            
            //renderHeatmapAvg( ctx->mlp_act, ract, fb, lut, true, clip_abs );
            
            //renderHeatmapAvg( ctx->mlp_l2, r2, fb, lut, true, clip_abs );
        }
    };
}