module;
#include <stdexcept>

export module Compute.OperationType;

export namespace Mila::Dnn::Compute
{
	export enum class OperationType {
		LayerNormOp,
		MatMulOp,
		SoftmaxOp
	};

	export std::string operationTypeToString( OperationType op ) {
		switch ( op ) {
			case OperationType::LayerNormOp: return "LayerNormOp";
			case OperationType::MatMulOp: return "MatMulOp";
			case OperationType::SoftmaxOp: return "SoftmaxOp";

			default:
				throw std::runtime_error( "Invalid OperationType." );
		}
	};
}
