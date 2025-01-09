module;
#include <iostream>
#include <set>
#include <string>

export module Compute.Operations;

export namespace Mila::Dnn::Compute
{
	export enum class Operation {
		LayerNorm,
		MatrixMultiply,
	};

	export std::string operationToString( Operation op ) {
		switch (op) {
		case Operation::LayerNorm: return "LayerNorm";
		case Operation::MatrixMultiply: return "MatrixMultiply";
		default:
			return "Unknown Op";
		}
	};
}
