module;
#include <stdexcept>

export module Compute.OperationType;

export namespace Mila::Dnn::Compute
{
	export enum class OperationType {
		kLayerNorm,
		kMatMul,
		kSoftmax
	};

	export std::string operationTypeToString( OperationType op ) {
		switch ( op ) {
			case OperationType::kLayerNorm: return "LayerNorm";
			case OperationType::kMatMul: return "MatMul";
			case OperationType::kSoftmax: return "Softmax";

			default:
				throw std::runtime_error( "Invalid OperationType." );
		}
	};
}
