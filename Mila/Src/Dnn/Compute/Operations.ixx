module;
#include <iostream>
#include <set>
#include <string>

export module Compute.Operations;

export namespace Mila::Dnn::Compute
{
	export enum class Operation {
		LayerNormOp,
		MatMulOp,
	};

	export std::string operationToString( Operation op ) {
		switch (op) {
		case Operation::LayerNormOp: return "LayerNormOp";
		case Operation::MatMulOp: return "MatMulOp";
		default:
			return "Unknown Operation.";
		}
	};
}
