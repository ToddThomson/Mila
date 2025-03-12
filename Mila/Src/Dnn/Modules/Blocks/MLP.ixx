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
import Compute.ComputeDevice;
import Compute.CpuDevice;

import Dnn.Modules.Linear;
import Dnn.Modules.Gelu;

namespace Mila::Dnn
{

    export
    template<typename TInput, typename TCompute = TInput, typename TDevice = Compute::CpuDevice>
        requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>
    class MLP : public Module<TInput, TCompute, TDevice> {
    public:
		using MR = TDevice::MR;

		MLP( std::string name, const std::vector<size_t>& input_shape, size_t output_channels, bool has_bias = true, bool is_training = false )
			: name_{ name }, input_shape_{ input_shape }, output_channels_{ output_channels }, is_training_ { is_training } {
			
			// Infer the number of input channels from the input shape
			input_channels_ = input_shape.back();

			// Construct the output shape for the fc_1_ layer
			std::vector<size_t> fc_1_output_shape = input_shape;
			fc_1_output_shape.back() = output_channels_;

			fc_1_ = std::make_unique<Linear<TInput, TCompute, TDevice>>( "fc_1", input_channels_, output_channels_ );
			gelu_ = std::make_unique<Gelu<TInput, TCompute, TDevice>>( "act_1" );
			fc_proj_ = std::make_unique<Linear<TInput, TCompute, TDevice>>( "fc_proj", output_channels_, output_channels_ );

			// Add sub modules
			/*addModule( "fc_1", fc_1_ );
			addModule( "gelu", gelu_ );
			addModule( "fc_proj", fc_proj_ );*/

			// Allocate output tensors for the MLP layers
			fc_1_output_ = Tensor<TCompute, MR>( fc_1_output_shape );
			gelu_output_ = Tensor<TCompute, MR>( fc_1_output_shape );
		}

		const std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>>& getSubModules() const override {
			//static std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>> subModules( { fc_1_, gelu_, fc_proj_ } );
			return {};
		}

		const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getParameters() const override {
			return {};
		}

		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output ) override {
			fc_1_->forward( input, fc_1_output_ );
			//fc_1_output_.print();
			gelu_->forward( fc_1_output_, gelu_output_ );
			//gelu_output_.print();
			fc_proj_->forward( gelu_output_, output );
			//output.print();
		}

		size_t parameterCount() const override {
			return fc_1_->parameterCount() + gelu_->parameterCount() + fc_proj_->parameterCount();
		}

		std::string name() const override {
			return name_;
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the child modules
			for ( const auto& [name, module] : this->sub_modules_ ) {
				module->save( zip );
			}

			//// Save the state of the parameters
			//for ( const auto& [name, tensor] : this->parameters_ ) {
			//	// Save tensor data to zip archive
			//}
		}

		void load( mz_zip_archive& zip ) override {
			// Load the state of the child modules
			for ( const auto& [name, module] : this->sub_modules_ ) {
				module->load( zip );
			}

			//// Load the state of the parameters
			//for ( const auto& [name, tensor] : this->parameters_ ) {
			//	// Load tensor data from zip archive
			//}
		}

		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameter count: " << parameterCount() << std::endl;
		}

    private:
        std::string name_; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t input_channels_; ///< The number of input channels
		size_t output_channels_;
		bool has_bias_{ true }; ///< Whether the module has a bias tensor. Default is true.
		bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.

		std::unique_ptr<Linear<TInput, TCompute, TDevice>> fc_1_{ nullptr };
		std::unique_ptr<Gelu<TInput, TCompute, TDevice>> gelu_{ nullptr };
		std::unique_ptr<Linear<TInput, TCompute, TDevice>> fc_proj_{ nullptr };

		Tensor<TCompute, MR> fc_1_output_;
		Tensor<TCompute, MR> gelu_output_;
    };
}