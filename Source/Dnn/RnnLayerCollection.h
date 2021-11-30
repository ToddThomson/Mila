#ifndef MILA_DNN_RNN_LAYER_COLLECTION_H_
#define MILA_DNN_RNN_LAYER_COLLECTION_H_

#include <memory>

#include <cudnn.h>

#include "RnnLinearLayer.h"

namespace Mila::Dnn
{
    class RnnLayerCollection
    {
    public:

        RnnLayerCollection()
        {
            layers_ = std::vector<RnnLinearLayer>();
        }

        void Add( RnnLinearLayer& layer )
        {
            layers_.push_back( std::move( layer ) );
        }

        //~RnnLayerCollection() = default;

    private:

        std::vector<RnnLinearLayer> layers_;
    };
}
#endif