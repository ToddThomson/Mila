module;
#include <random>
#include <cmath>

export module Dnn.TensorHelpers;

import Dnn.Tensor;

namespace Mila::Dnn
{
	export
		void random( Tensor<float>& tensor, float min, float max ) {
		std::random_device rd;
		std::mt19937 gen( rd() );
		std::uniform_real_distribution<float> dis( min, max );

		auto v = tensor.vectorSpan();
		for ( size_t i = 0; i < tensor.size(); ++i ) {
			v[ i ] = dis( gen );
		}
	}

	export void xavier( Tensor<float>& tensor, size_t input_size, size_t output_size ) {
		float limit = std::sqrt( 6.0 / (input_size + output_size) );
		std::random_device rd;
		std::mt19937 gen( rd() );
		std::uniform_real_distribution<float> dis( -limit, limit );

		for ( size_t i = 0; i < tensor.size(); ++i ) {
			tensor[ i ] = dis( gen );
		}
	}
}