module;
#include <string>
#include <stdexcept>

export module Compute.OperationType;

export namespace Mila::Dnn::Compute
{
	export enum class OperationType {
		AttentionOp,
		CrossEntropyOp,
		EncoderOp,
		GeluOp,
		LayerNormOp,
		MatMulOp,
		ResidualOp,
		SoftmaxOp
	};

	export std::string operationTypeToString( OperationType op ) {
		switch ( op ) {
			case OperationType::AttentionOp: return "AttentionOp";
			case OperationType::CrossEntropyOp: return "CrossEntropyOp";
			case OperationType::EncoderOp: return "EncoderOp";
			case OperationType::GeluOp: return "GeluOp";
			case OperationType::LayerNormOp: return "LayerNormOp";
			case OperationType::MatMulOp: return "MatMulOp";
			case OperationType::ResidualOp: return "ResidualOp";
			case OperationType::SoftmaxOp: return "SoftmaxOp";

			default:
				throw std::runtime_error( "Invalid OperationType." );
		}
	};
}