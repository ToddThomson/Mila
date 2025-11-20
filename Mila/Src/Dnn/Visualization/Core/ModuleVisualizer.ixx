/* ---------------------------
   Base class for module visualizers
   --------------------------- */
module;

export module Visualization.Model;

import Visualization.FrameBuffer;

namespace Mila::Dnn::Visualization
{
    export class ModuleVisualizer
    {
    public:
        
        virtual ~ModuleVisualizer()
        {
        }
        
        virtual void render( Framebuffer& fb, const Rect& region ) = 0;
    };
}