/**
 * @file MilaModules.ixx
 * @brief Aggregate module that re-exports Mila built-in DNN modules.
 */

module;
export module Modules.MilaModules;

// Activations
export import Dnn.Components.Gelu;

// Blocks
export import Dnn.Blocks.MLP;
export import Dnn.Blocks.Transformer;

// Connection Layers
export import Dnn.Components.Residual;

// Layers
export import Dnn.Components.Attention;
export import Dnn.Components.Encoder;
export import Dnn.Components.Linear;

//Losses
export import Dnn.Components.SoftmaxCrossEntropy;
export import Dnn.Components.Softmax;

// Normalization
export import Dnn.Components.LayerNorm;

// Regularization
// FUTURE: export import Dnn.Components.Dropout;