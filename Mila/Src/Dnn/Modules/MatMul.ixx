module;
#include <memory>
#include <vector>
#include <string>

export module Dnn.Modules.MatMul;

import Dnn.Module;
import Compute.OperationBase;
import Compute.OperationsFactory;

export namespace Mila::Dnn::Modules
{
	export template<typename T>
	class MatMul : public Module<T>{
    public:
        MatMul( const std::string& deviceType ) {
            operation_ = Compute::OperationsFactory::createOperation( deviceType, "MatMulOp" );
        }

        std::string name() const override {
            return "MatMulModule";
        }

        /*void forward( const std::vector<float>& inputs, std::vector<float>& outputs ) const override {
            operation_->forward( inputs, outputs );
        }*/

        void backward( const std::vector<float>& grad_outputs, std::vector<float>& grad_inputs ) const override {
            operation_->backward( grad_outputs, grad_inputs );
        }

    private:
        std::shared_ptr<Dnn::Compute::OperationBase<T>> operation_;
    };
}