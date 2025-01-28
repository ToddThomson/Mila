module;
#include <stdexcept>

export module Compute.OperationType;

export namespace Mila::Dnn::Compute
{
	export enum class OperationType {
		kLayerNormOp,
		kMatMulOp,
	};

	export std::string operationTypeToString( OperationType op ) {
		switch (op) {
		case OperationType::kLayerNormOp: return "LayerNormOp";
		case OperationType::kMatMulOp: return "MatMulOp";
		default:
			throw std::runtime_error( "Invalid OperationType." );
		}
	};
}
