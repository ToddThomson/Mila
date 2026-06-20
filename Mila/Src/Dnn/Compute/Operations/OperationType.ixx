/**
 * @file OperationType.ixx
 * @brief Defines the operation types supported by the compute framework.
 *
 * ARCHITECTURAL NOTE (TODO):
 * OperationType is an internal dispatch key used by the compute layer. It is not
 * part of the public Mila API — ComponentType (Dnn.ComponentType) is the user-facing
 * component identity. OperationType should be moved to Dnn::Core and removed from
 * the public Mila.ixx re-exports so it is inaccessible to library consumers.
 * Operations are an implementation detail of Components; users should never need
 * to reference OperationType directly.
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
		CrossEntropyOp,				///< Cross entropy loss operation (host-based; used by GPT reference implementation)
		TokenEmbeddingOp,			///< Token embedding operation
		LpeOp,						///< Learned Positional Embedding operation for transformer architecture
        RopeOp,						///< Rotary Position Embedding operation for transformer architecture
		FusedOp,					///< Fused operation combining multiple operations for performance optimization
		LinearOp,					///< Linear (fully connected/dense) layer operation
		GeluOp,						///< Gaussian Error Linear Unit activation function
        ElementwiseActivationOp,	///< Functor-templated elementwise activation (GELU/SiLU/ReLU/Tanh/Sigmoid/LeakyReLU/Mish)
        SwigluOp,					///< SwiGLU (SiLU-gated) GLU FFN activation
        GegluOp,					///< GeGLU (GELU-gated) GLU FFN activation — Gemma
		LayerNormOp,				///< Layer normalization operation
        RmsNormOp,					///< RMS normalization operation
		MultiHeadAttentionOp,		///< Multi-head attention operation (MHA) for transformers
		GroupedQueryAttentionOp,	///< Grouped Query Attention (GQA)
		ResidualOp,					///< Residual connection operation
		SoftmaxOp,					///< Softmax activation function
		DropoutOp,					///< Dropout regularization operation
		SamplingOp,					///< Device-side token sampling from logits
		SoftmaxCrossEntropyOp		///< WIP: Fused softmax + cross-entropy loss — targeted for Llama training
	};

	// string_view constants, no magic strings at call sites
	export namespace OperationNames
	{
		constexpr std::string_view CrossEntropy         = "CrossEntropyOp";
		constexpr std::string_view TokenEmbedding       = "TokenEmbeddingOp";
		constexpr std::string_view Lpe                  = "LpeOp";
		constexpr std::string_view Rope                 = "RopeOp";
		constexpr std::string_view Fused                = "FusedOp";
		constexpr std::string_view Linear               = "LinearOp";
		constexpr std::string_view Gelu                 = "GeluOp";
		constexpr std::string_view ElementwiseActivation = "ElementwiseActivationOp";
		constexpr std::string_view Swiglu               = "SwigluOp";
		constexpr std::string_view Geglu                = "GegluOp";
		constexpr std::string_view LayerNorm            = "LayerNormOp";
		constexpr std::string_view RmsNorm              = "RmsNormOp";
		constexpr std::string_view MultiHeadAttention   = "MultiHeadAttentionOp";
		constexpr std::string_view GroupedQueryAttention = "GroupedQueryAttentionOp";
		constexpr std::string_view Residual             = "ResidualOp";
		constexpr std::string_view Softmax              = "SoftmaxOp";
		constexpr std::string_view Dropout              = "DropoutOp";
		constexpr std::string_view Sampling             = "SamplingOp";
		constexpr std::string_view SoftmaxCrossEntropy  = "SoftmaxCrossEntropyOp"; ///< WIP — targeted for Llama training
	}

	export std::string_view operationTypeToString( OperationType op )
	{
		switch ( op )
		{
			case OperationType::CrossEntropyOp:          return OperationNames::CrossEntropy;
			case OperationType::TokenEmbeddingOp:        return OperationNames::TokenEmbedding;
			case OperationType::LpeOp:                   return OperationNames::Lpe;
			case OperationType::RopeOp:                  return OperationNames::Rope;
			case OperationType::FusedOp:                 return OperationNames::Fused;
			case OperationType::LinearOp:                return OperationNames::Linear;
			case OperationType::GeluOp:                  return OperationNames::Gelu;
			case OperationType::ElementwiseActivationOp: return OperationNames::ElementwiseActivation;
			case OperationType::SwigluOp:                return OperationNames::Swiglu;
			case OperationType::GegluOp:                 return OperationNames::Geglu;
			case OperationType::LayerNormOp:             return OperationNames::LayerNorm;
			case OperationType::RmsNormOp:               return OperationNames::RmsNorm;
			case OperationType::MultiHeadAttentionOp:    return OperationNames::MultiHeadAttention;
			case OperationType::GroupedQueryAttentionOp: return OperationNames::GroupedQueryAttention;
			case OperationType::ResidualOp:              return OperationNames::Residual;
			case OperationType::SoftmaxOp:               return OperationNames::Softmax;
			case OperationType::DropoutOp:               return OperationNames::Dropout;
			case OperationType::SamplingOp:              return OperationNames::Sampling;
			case OperationType::SoftmaxCrossEntropyOp:   return OperationNames::SoftmaxCrossEntropy;
			default:
				throw std::runtime_error( "operationTypeToString: unrecognized OperationType" );
		}
	}
}