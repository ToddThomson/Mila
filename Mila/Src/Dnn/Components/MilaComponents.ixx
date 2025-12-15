/**
 * @file MilaComponents.ixx
 * @brief Aggregate module that re-exports Mila built-in DNN components.
 */

module;
export module Components.MilaComponents;

// Activations
export import Dnn.Components.Gelu;

// Blocks
export import Dnn.Blocks.MLP;
// FIXME: export import Dnn.Blocks.Transformer;

// Connection Layers
// FIXME: export import Dnn.Components.Residual;

// Layers
// FIXME: export import Dnn.Components.Attention;
// export import Dnn.Components.Encoder;
export import Dnn.Components.Linear;

//Losses
// FIXME: export import Dnn.Components.SoftmaxCrossEntropy;
export import Dnn.Components.Softmax;

// Normalization
export import Dnn.Components.LayerNorm;

// Regularization
// FUTURE: export import Dnn.Components.Dropout;