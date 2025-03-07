module;
#include <random>
#include <cmath>
#include <type_traits>

export module Dnn.TensorHelpers;

import Dnn.Tensor;
import Dnn.TensorTraits;

import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
	export template<typename T, typename MR = Compute::CpuMemoryResource> 
		requires std::is_base_of_v<Compute::MemoryResource, MR>
	void random( Tensor<float, MR>& tensor, float min, float max ) {
		std::random_device rd;
		std::mt19937 gen( 42 ); // rd() );
		std::uniform_real_distribution<float> dis( min, max );

		if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
			auto temp = tensor.to<Compute::CpuMemoryResource>();
			auto v = temp.vectorSpan();
			for ( size_t i = 0; i < temp.size(); ++i ) {
				temp[ i ] = dis( gen );
			}
			tensor = temp.to<MR>();
		}
		else {
			auto v = tensor.vectorSpan();
			for ( size_t i = 0; i < tensor.size(); ++i ) {
				v[ i ] = dis( gen );
			}
		}
	}

	export template<typename T, typename MR> requires std::is_base_of_v<Compute::MemoryResource, MR>
	void xavier( Tensor<float, MR>& tensor, size_t input_size, size_t output_size ) {
		float limit = std::sqrt( 6.0 / (input_size + output_size) );
		std::random_device rd;
		std::mt19937 gen( 42 );// rd() );
		std::uniform_real_distribution<float> dis( -limit, limit );

		if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
			auto temp = tensor.to<Compute::CpuMemoryResource>();
			auto v = temp.vectorSpan();
			for ( size_t i = 0; i < temp.size(); ++i ) {
				v[ i ] = dis( gen );
			}
			tensor = temp.to<MR>();
		}
		else {
			auto v = tensor.vectorSpan();
			for ( size_t i = 0; i < tensor.size(); ++i ) {
				v[ i ] = dis( gen );
			}
		}
	}
}