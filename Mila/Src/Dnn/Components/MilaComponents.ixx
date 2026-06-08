/**
 * @file MilaComponents.ixx
 * @brief Aggregate module that re-exports Mila built-in DNN components.
 */

module;
export module Components.MilaComponents;

// Activations
export import Dnn.Components.Gelu;
export import Dnn.Components.Swiglu;

// Composite Layers
export import Dnn.Components.MLP;
export import Dnn.Components.GptBlock;

// Connection Layers
export import Dnn.Components.Residual;

// Layers
export import Dnn.Components.MultiHeadAttention;
export import Dnn.Components.Lpe;
export import Dnn.Components.Rope;
export import Dnn.Components.Linear;

//Losses
// FIXME: export import Dnn.Components.SoftmaxCrossEntropy;

// Normalization
export import Dnn.Components.LayerNorm;
export import Dnn.Components.Softmax;

// Regularization
// FUTURE: export import Dnn.Components.Dropout;