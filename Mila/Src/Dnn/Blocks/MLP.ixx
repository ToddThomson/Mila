module;
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

export module Dnn.Blocks.MLP;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

import Dnn.Modules.Linear;
import Dnn.Modules.Gelu;

namespace Mila::Dnn::Blocks
{
    using namespace Mila::Dnn;

    export
    template<typename TInput, typename TCompute = TInput, typename MR = Compute::CpuMemoryResource>
        requires ValidTensorTypes<TInput, TCompute>&& std::is_base_of_v<Compute::MemoryResource, MR>
    class MLP : public Module<TInput, TCompute, MR> {
    public:
        MLP() {
			auto shape = std::vector<size_t>{ 4, 4, 4 };
			auto output_channels = 4;

            fc_1_ = std::make_unique<Modules::Linear<TInput,TCompute, MR>>( "fc_1", shape, output_channels );
            gelu_ = std::make_unique<Modules::Gelu<TInput,TCompute,MR>>( "act_1", shape );
            fc_proj_ = std::make_unique<Modules::Linear<TInput,TCompute,MR>>( "fc_proj", shape, output_channels );
        }

		Tensor<TCompute, MR>&& forward( const Tensor<TInput, MR>& input ) {
			auto x = fc_1_->forward( input );
			auto y = gelu_->forward( x );
			auto z = fc_proj_->forward( y );
			return std::move( z );
		}

		size_t parameters() const override {
			return fc_1_->parameters() + gelu_->parameters() + fc_proj_->parameters();
		}

		std::string name() const override {
			return name_;
		}

		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameters: " << parameters() << std::endl;
		}

    private:
        std::string name_; ///< The name of the module.
        std::vector<size_t> input_shape_; ///< The input shape.

		std::unique_ptr<Modules::Linear<TInput, TCompute, MR>> fc_1_{ nullptr };
		std::unique_ptr<Modules::Gelu<TInput, TCompute, MR>> gelu_{ nullptr };
		std::unique_ptr<Modules::Linear<TInput, TCompute, MR>> fc_proj_{ nullptr };
    };
}