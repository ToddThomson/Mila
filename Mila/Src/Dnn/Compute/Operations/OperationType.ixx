/**
 * @file OperationType.ixx
 * @brief Defines the operation types supported by the compute framework.
 */

module;
#include <string>
#include <stdexcept>

export module Compute.OperationType;

namespace Mila::Dnn::Compute
{
	/**
	 * @brief Enumeration of all supported neural network operation types.
	 *
	 * This enumeration defines the different types of operations that can be
	 * executed by the compute framework. Each operation type corresponds to
	 * a specific neural network function or layer.
	 */
	export enum class OperationType {
		CrossEntropyOp,				///< Cross entropy loss operation
		TokenEmbeddingOp,			///< Token embedding operation
		LpeOp,						///< Learned Positional Embedding operation for transformer architecture
        RopeOp,						///< Rotary Position Embedding operation for transformer architecture
		FusedOp,					///< Fused operation combining multiple operations for performance optimization
		LinearOp,					///< Linear (fully connected/dense) layer operation
		GeluOp,						///< Gaussian Error Linear Unit activation function
        SwigluOp,					///< SwiGLU activation function
		LayerNormOp,				///< Layer normalization operation
        RmsNormOp,					///< RMS normalization operation
		MultiHeadAttentionOp,		///< Multi-head attention operation (MHA) for transformers
		GroupedQueryAttentionOp,	///< Groupted Query Attention (GQA)
		ResidualOp,					///< Residual connection operation
		SoftmaxOp					///< Softmax activation function
	};

	// string_view constants, no magic strings at call sites
	export namespace OperationNames
	{
		constexpr std::string_view CrossEntropy = "CrossEntropyOp";
		constexpr std::string_view Lpe = "LpeOp";
		constexpr std::string_view Rope = "RopeOp";
		constexpr std::string_view Fused = "FusedOp";
		constexpr std::string_view Linear = "LinearOp";
		constexpr std::string_view Gelu = "GeluOp";
		constexpr std::string_view Swiglu = "SwigluOp";
		constexpr std::string_view LayerNorm = "LayerNormOp";
		constexpr std::string_view RmsNorm = "RmsNormOp";
		constexpr std::string_view MultiHeadAttention = "MultiHeadAttentionOp";
		constexpr std::string_view GroupedQueryAttention = "GroupedQueryAttentionOp";
		constexpr std::string_view Residual = "ResidualOp";
		constexpr std::string_view Softmax = "SoftmaxOp";
	}

	export std::string_view operationTypeToString( OperationType op )
	{
		switch ( op )
		{
			case OperationType::CrossEntropyOp:          return OperationNames::CrossEntropy;
			case OperationType::LpeOp:                   return OperationNames::Lpe;
			case OperationType::RopeOp:                  return OperationNames::Rope;
			case OperationType::FusedOp:                 return OperationNames::Fused;
			case OperationType::LinearOp:                return OperationNames::Linear;
			case OperationType::GeluOp:                  return OperationNames::Gelu;
			case OperationType::SwigluOp:                return OperationNames::Swiglu;
			case OperationType::LayerNormOp:             return OperationNames::LayerNorm;
			case OperationType::RmsNormOp:               return OperationNames::RmsNorm;
			case OperationType::MultiHeadAttentionOp:    return OperationNames::MultiHeadAttention;
			case OperationType::GroupedQueryAttentionOp: return OperationNames::GroupedQueryAttention;
			case OperationType::ResidualOp:              return OperationNames::Residual;
			case OperationType::SoftmaxOp:               return OperationNames::Softmax;
			default:
				throw std::runtime_error( "operationTypeToString: unrecognized OperationType" );
		}
	}
}