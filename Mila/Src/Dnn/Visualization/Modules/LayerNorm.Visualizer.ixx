export module Visualization.LayerNorm;

/* ---------------------------
   LayerNorm Visualizer
   ---------------------------
*/

import Visualization.Model;
import Visualization.Context;
import Visualization.FrameBuffer;
import Visualization.ColorLUT;
import Visualization.HeatmapRenderer;

namespace Mila::Dnn::Visualization
{
    export class LayerNormVisualizer : ModuleVisualizer
    {
    private:
        VisualizerContext* ctx;
        const ColorLUT& lut;
        float clip_abs;
    public:
        
        LayerNormVisualizer( VisualizerContext* c, const ColorLUT& l, float clip = 0.0f ) 
            : ctx( c ), lut( l ), clip_abs( clip )
        {
        }
        
        void render( Framebuffer& fb, const Rect& region ) override
        {
            // Render ln_out (tokens x hidden) as heatmap token x channel
            //renderHeatmapAvg( ctx->ln_out, region, fb, lut, true, clip_abs );
        }
    };
}