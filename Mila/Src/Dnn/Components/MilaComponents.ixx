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
export import Dnn.Components.LlamaBlock;

// Connection Layers
export import Dnn.Components.Residual;

// Layers
export import Dnn.Components.Attention;
export import Dnn.Components.LearnedEncoder;
export import Dnn.Components.RopeEncoder;
export import Dnn.Components.Linear;

//Losses
// FIXME: export import Dnn.Components.SoftmaxCrossEntropy;

// Normalization
export import Dnn.Components.LayerNorm;
export import Dnn.Components.Softmax;

// Regularization
// FUTURE: export import Dnn.Components.Dropout;