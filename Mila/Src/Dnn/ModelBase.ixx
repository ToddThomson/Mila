export module Dnn.ModelBase;

namespace Mila::Dnn
{
    export class ModelBase {
    public:
        virtual ~ModelBase() = default;

        virtual void print() const = 0;
    };
}