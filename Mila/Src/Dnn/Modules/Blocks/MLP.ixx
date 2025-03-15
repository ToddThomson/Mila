module;
#include <iostream>
#include <sstream>
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
			: input_shape_{ input_shape }, output_channels_{ output_channels } {
			this->setName( name );
			this->setTraining( is_training );

			// Infer the number of input channels from the input shape
			input_channels_ = input_shape.back();

			// Construct the output shape for the fc_1_ layer
			std::vector<size_t> fc_1_output_shape = input_shape;
			fc_1_output_shape.back() = output_channels_;

			fc_1_ = std::make_shared<Linear<TInput, TCompute, TDevice>>( this->getName() + ".fc_1", input_channels_, output_channels_ );
			gelu_ = std::make_shared<Gelu<TInput, TCompute, TDevice>>( this->getName() + ".gelu" );
			fc_proj_ = std::make_shared<Linear<TInput, TCompute, TDevice>>( this->getName() + ".fc_proj", output_channels_, input_channels_ );

			// Add sub-modules to the MLP block
			this->addModule( fc_1_ );
			this->addModule( gelu_ );
			this->addModule( fc_proj_ );

			// Allocate output tensors for the MLP layers
			fc_1_output_ = Tensor<TCompute, MR>( fc_1_output_shape );
			gelu_output_ = Tensor<TCompute, MR>( fc_1_output_shape );
		}

		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute, MR>& output ) override {
			fc_1_->forward( input, fc_1_output_ );
			gelu_->forward( fc_1_output_, gelu_output_ );
			fc_proj_->forward( gelu_output_, output );
		}

		size_t parameterCount() const override {
			size_t total_parameters = 0;
			for ( const auto& module : this->getSubModules() ) {
				total_parameters += module->parameterCount();
			}
			return total_parameters;
		}

		void save( mz_zip_archive& zip ) const override {
			for ( const auto& module : this->getSubModules() ) {
				module->save( zip );
			}

			//// Save the state of the parameters
			//for ( const auto& [name, tensor] : this->parameters_ ) {
			//	// Save tensor data to zip archive
			//}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& module : this->getSubModules() ) {
				module->load( zip );
			}

			//// Load the state of the parameters
			//for ( const auto& [name, tensor] : this->parameters_ ) {
			//	// Load tensor data from zip archive
			//}
		}

        std::string toString() const override {
			std::ostringstream oss;
			oss << "====================" << std::endl;
			oss << "MLP: " << this->getName();
			oss << ", Input shape: (";
			for ( size_t i = 0; i < input_shape_.size(); ++i ) {
				oss << input_shape_[ i ];
				if ( i != input_shape_.size() - 1 ) {
					oss << ",";
				}
			}
			oss << ")";
			oss << ", Input channels: " << input_channels_;
			oss << ", Output channels: " << output_channels_ << std::endl;
			oss << "Parameter count: " << parameterCount() << std::endl;
			oss << "Sub-Modules..." << std::endl;
            for (const auto& module : this->getSubModules()) {
                oss << *module;
            }
			
			return oss.str();
        }

    private:
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t input_channels_; ///< The number of input channels
		size_t output_channels_; ///< The number of output channels
		//bool has_bias_{ true }; ///< Whether the module has a bias tensor. Default is true.

		std::shared_ptr<Linear<TInput, TCompute, TDevice>> fc_1_{ nullptr };
		std::shared_ptr<Gelu<TInput, TCompute, TDevice>> gelu_{ nullptr };
		std::shared_ptr<Linear<TInput, TCompute, TDevice>> fc_proj_{ nullptr };

		Tensor<TCompute, MR> fc_1_output_;
		Tensor<TCompute, MR> gelu_output_;
    };
}