module;
#include <vector>
#include <string>
#include <functional>
#include <memory>


export module Compute.Operation;

namespace Mila::Dnn::Compute
{
    class IOperation {
    public:
        virtual ~IOperation() = default;
    };

    template <typename OpType>
    class OperationWrapper : public IOperation {
    public:
        using Creator = std::function<std::unique_ptr<OpType>()>;

    private:
        Creator creator;

    public:
        OperationWrapper( Creator creator ) : creator( std::move( creator ) ) {}

        std::unique_ptr<OpType> create() const {
            return creator();
        }
    };
}