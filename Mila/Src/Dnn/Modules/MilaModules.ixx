/**
 * @file MilaModules.ixx
 * @brief Aggregate module that re-exports Mila built-in DNN modules.
 */

module;
export module Modules.MilaModules;

// Activations
export import Dnn.Modules.Gelu;

// Blocks
export import Dnn.Blocks.MLP;
export import Dnn.Blocks.Transformer;

// Connection Layers
export import Dnn.Modules.Residual;

// Layers
export import Dnn.Modules.Attention;
export import Dnn.Modules.Encoder;
export import Dnn.Modules.Linear;

//Losses
export import Dnn.Modules.SoftmaxCrossEntropy;
export import Dnn.Modules.Softmax;

// Normalization
export import Dnn.Modules.LayerNorm;

// Regularization
// FUTURE: export import Dnn.Modules.Dropout;