/* ---------------------------
   Composed Block Visualizer
   Accepts a set of module visualizers, arranges them into master layout
   --------------------------- */
module;

export module Visualization.Transformer;

import Visualization.Context;
import Visualization.FrameBuffer;
import Visualization.ColorLUT;

namespace Mila::Dnn::Visualization
{
    export class BlockVisualizer
    {
	private:
        
        //LVisualizer ln;
        //MHAVisualizer mha;
        //MLPVisualizer mlp;
        //ResidualVisualizer resid;
        Framebuffer& fb;
        
	public:

        BlockVisualizer( VisualizerContext* ctx, Framebuffer& fb_, const ColorLUT& lut )
            : /* ln(ctx, lut, 3.0f), mha(ctx, lut, 3.0f), mlp(ctx, lut, 3.0f), resid(ctx), */ fb(fb_)
        {
        }

        void renderAll()
        {
            /* FIXME:
            * 
            // Entire frame: we'll create a grid matching the earlier plan:
            // Row1 -> LN_out + Q K V
            // Row2 -> Attention maps (large)
            // Row3 -> MLP triple
            // Row4 -> Residual strip small
            int W = fb.width, H = fb.height;
            int row1_h = 300;
            int row2_h = 1100;
            int row3_h = 600;
            int row4_h = H - (row1_h + row2_h + row3_h);
            

            // Row1: LN (left 25%), then Q K V split remaining 75%? earlier design used LN full width,
            // but we'll do LN full width left 40% and Q/K/V each 20%
            int y0 = 0;
            Rect rLN{ 0, y0, int( W * 0.4 ), row1_h };
            Rect rQ{ int( W * 0.4 ), y0, int( W * 0.2 ), row1_h };
            Rect rK{ int( W * 0.6 ), y0, int( W * 0.2 ), row1_h };
            Rect rV{ int( W * 0.8 ), y0, W - int( W * 0.8 ), row1_h };
            
            ln.render( fb, rLN );
            
            // quick copy q/k/v into mha expected regions
            // we'll render Q/K/V inside mha's top area too (mha arranges internally) - so we prepare ctx.q_proj etc.
            // Row2
            int y1 = y0 + row1_h;
            Rect rAtt{ 0, y1, W, row2_h };
            
            mha.render( fb, rAtt );
            
            // Row3
            int y2 = y1 + row2_h;
            Rect rMLP{ 0, y2, W, row3_h };
            mlp.render( fb, rMLP );
            
            // Row4 residual
            int y3 = y2 + row3_h;
            Rect rRes{ 0, y3, W, row4_h };
            
            resid.render( fb, rRes );
            */
        }
    };
}