/**
 * @file OperationType.ixx
 * @brief Defines the operation types supported by the compute framework.
 */

module;
#include <string>
#include <stdexcept>

export module Compute.OperationType;

export namespace Mila::Dnn::Compute
{
	/**
	 * @brief Enumeration of all supported neural network operation types.
	 *
	 * This enumeration defines the different types of operations that can be
	 * executed by the compute framework. Each operation type corresponds to
	 * a specific neural network function or layer.
	 */
	export enum class OperationType {
		CrossEntropyOp,        ///< Cross entropy loss operation
		EncoderOp,             ///< Encoder operation for transformer architecture
		FusedOp,               ///< Fused operation combining multiple operations for performance optimization
		FullyConnectedOp,      ///< Fully connected (dense) layer operation
		GeluOp,                ///< Gaussian Error Linear Unit activation function
		LayerNormOp,           ///< Layer normalization operation
		MultiHeadAttentionOp,  ///< Multi-head attention operation for transformers
		ResidualOp,            ///< Residual connection operation
		SoftmaxOp              ///< Softmax activation function
	};

	/**
	 * @brief Converts an operation type to its string representation.
	 *
	 * This utility function converts an OperationType enum value to a human-readable
	 * string representation, which can be used for logging, debugging, or serialization.
	 *
	 * @param op The operation type to convert to string
	 * @return std::string The string representation of the operation type
	 * @throws std::runtime_error If the operation type is invalid or not recognized
	 */
	export std::string operationTypeToString( OperationType op ) {
		switch ( op ) {
			case OperationType::CrossEntropyOp: return "CrossEntropyOp";
			case OperationType::EncoderOp: return "EncoderOp";
			case OperationType::FusedOp: return "FusedOp";
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
