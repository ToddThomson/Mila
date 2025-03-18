module;
#include <string>
#include <stdexcept>

export module Compute.OperationType;

export namespace Mila::Dnn::Compute
{
	export enum class OperationType {
		CrossEntropyOp,
		EncoderOp,
		FullyConnectedOp,
		GeluOp,
		LayerNormOp,
		MultiHeadAttentionOp,
		ResidualOp,
		SoftmaxOp
	};

	export std::string operationTypeToString( OperationType op ) {
		switch ( op ) {
			case OperationType::CrossEntropyOp: return "CrossEntropyOp";
			case OperationType::EncoderOp: return "EncoderOp";
			case OperationType::FullyConnectedOp: return "FullyConnectedOp";
			case OperationType::GeluOp: return "GeluOp";
			case OperationType::LayerNormOp: return "LayerNormOp";
			case OperationType::MultiHeadAttentionOp: return "MultiHeadAttentionOp";
			case OperationType::ResidualOp: return "ResidualOp";
			case OperationType::SoftmaxOp: return "SoftmaxOp";

			default:
				throw std::runtime_error( "Invalid OperationType." );
		}
	};
}