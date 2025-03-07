module;
#include <corecrt_math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>

export module Gpt2.Gpt2Model;

import Mila;
import Gpt2.Gpt2Config;

namespace Mila::Dnn::Gpt2
{
	using namespace Mila::Dnn::Compute;

	constexpr int Gpt2ModelHeaderSize = 256;
	constexpr int NumberOfParameterTensors = 16;

	/**
	 * @struct ParameterTensors
	 * @brief Holds the parameter tensors for the GPT-2 model.
	 */
	struct ParameterTensors {
		std::vector<float> wte; ///< Token embedding weights (V, C)
		std::vector<float> wpe; ///< Position embedding weights (maxT, C)

		std::vector<float> ln1w; ///< Layer normalization 1 weights (L, C)
		std::vector<float> ln1b; ///< Layer normalization 1 biases (L, C)

		std::vector<float> qkvw; ///< Query, Key, Value weights (L, 3*C, C)
		std::vector<float> qkvb; ///< Query, Key, Value biases (L, 3*C)

		std::vector<float> attprojw; ///< Attention projection weights (L, C, C)
		std::vector<float> attprojb; ///< Attention projection biases (L, C)

		std::vector<float> ln2w; ///< Layer normalization 2 weights (L, C)
		std::vector<float> ln2b; ///< Layer normalization 2 biases (L, C)

		std::vector<float> fcw; ///< Fully connected layer weights (L, 4*C, C)
		std::vector<float> fcb; ///< Fully connected layer biases (L, 4*C)

		std::vector<float> fcprojw; ///< Fully connected projection weights (L, C, 4*C)
		std::vector<float> fcprojb; ///< Fully connected projection biases (L, C)

		std::vector<float> lnfw; ///< Final layer normalization weights (C)
		std::vector<float> lnfb; ///< Final layer normalization biases (C)
	};

	constexpr int NumberOfActivationTensors = 23;

	/**
	 * @struct ActivationTensors
	 * @brief Holds the activation tensors for the GPT-2 model.
	 */
	struct ActivationTensors {
		std::vector<float> encoded; ///< Encoded input tokens (B, T, C)
		std::vector<float> ln1; ///< Layer normalization 1 activations (L, B, T, C)
		std::vector<float> ln1_mean; ///< Layer normalization 1 mean (L, B, T)
		std::vector<float> ln1_rstd; ///< Layer normalization 1 reciprocal standard deviation (L, B, T)
		std::vector<float> qkv; ///< Query, Key, Value activations (L, B, T, 3*C)
		std::vector<float> atty; ///< Attention output (L, B, T, C)
		std::vector<float> preatt; ///< Pre-attention activations (L, B, NH, T, T)
		std::vector<float> att; ///< Attention activations (L, B, NH, T, T)
		std::vector<float> attproj; ///< Attention projection activations (L, B, T, C)
		std::vector<float> residual2; ///< Residual connection 2 activations (L, B, T, C)
		std::vector<float> ln2; ///< Layer normalization 2 activations (L, B, T, C)
		std::vector<float> ln2_mean; ///< Layer normalization 2 mean (L, B, T)
		std::vector<float> ln2_rstd; ///< Layer normalization 2 reciprocal standard deviation (L, B, T)
		std::vector<float> fch; ///< Fully connected hidden layer activations (L, B, T, 4*C)
		std::vector<float> fch_gelu; ///< GELU activations (L, B, T, 4*C)
		std::vector<float> fcproj; ///< Fully connected projection activations (L, B, T, C)
		std::vector<float> residual3; ///< Residual connection 3 activations (L, B, T, C)
		std::vector<float> lnf; ///< Final layer normalization activations (B, T, C)
		std::vector<float> lnf_mean; ///< Final layer normalization mean (B, T)
		std::vector<float> lnf_rstd; ///< Final layer normalization reciprocal standard deviation (B, T)
		std::vector<float> logits; ///< Logits (B, T, V)
		std::vector<float> probs; ///< Probabilities (B, T, V)
		std::vector<float> losses; ///< Losses (B, T)
	};

	export class Gpt2Model {
	public:
		Gpt2Model( const ModelConfig& config, size_t batch_size, size_t sequence_length, bool is_training = true )
			: config_( config ), batch_size_( batch_size ), seq_len_( sequence_length ), is_training_( is_training ) {
			initialize_parameter_tensor_sizes();
			initialize_activation_tensor_sizes();

			// Note: initialization of the activation tensors is handled during the current forward pass
			// TJT: The model should be initialized and ready upon completion of the constructor
		}

		float get_mean_loss() const {
			return mean_loss_;
		}

		const ActivationTensors& get_activations() const {
			return acts_;
		}

		const ModelConfig& get_config() const {
			return config_;
		}

		void fromCheckpoint( const std::string& checkpoint_path ) {
			// read in model from a checkpoint file
			std::ifstream model_file( checkpoint_path, std::ifstream::binary );
			if ( !model_file.is_open() ) {
				throw std::runtime_error( std::format( "Could not open model file: {}", checkpoint_path ) );
			}

			// read in the header
			std::array<int, Gpt2ModelHeaderSize> model_header;
			model_file.read( reinterpret_cast<char*>(&model_header), Gpt2ModelHeaderSize * sizeof( int ) );

			// check the magic number and version
			if ( model_header[ 0 ] != 20240326 ) {
				throw std::runtime_error( std::format( "Invalid magic number for model file: {}", checkpoint_path ) );
			}

			if ( model_header[ 1 ] != 3 ) {
				throw std::runtime_error( std::format( "Invalid version for model file: {}", checkpoint_path ) );
			}

			// read in hyperparameters, use size_t to prevent int overflow
			size_t maxT, V, Vp, L, NH, C;
			config_.max_seq_len = maxT = model_header[ 2 ];
			config_.vocab_size = V = model_header[ 3 ];
			config_.num_layers = L = model_header[ 4 ];
			config_.num_heads = NH = model_header[ 5 ];
			config_.channels = C = model_header[ 6 ];
			config_.padded_vocab_size = Vp = model_header[ 7 ];

			// allocate Tensors for all the parameters 
			initialize_parameter_tensors( params_ );

			// read in all the parameters from the checkpoint file
			// TODO: Move to separate code block
			model_file.read( reinterpret_cast<char*>(params_.wte.data()), params_.wte.size() * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.wpe[ 0 ]), param_sizes_[ 1 ] * sizeof( float ) );

			model_file.read( reinterpret_cast<char*>(&params_.ln1w[ 0 ]), param_sizes_[ 2 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.ln1b[ 0 ]), param_sizes_[ 3 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.qkvw[ 0 ]), param_sizes_[ 4 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.qkvb[ 0 ]), param_sizes_[ 5 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.attprojw[ 0 ]), param_sizes_[ 6 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.attprojb[ 0 ]), param_sizes_[ 7 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.ln2w[ 0 ]), param_sizes_[ 8 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.ln2b[ 0 ]), param_sizes_[ 9 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.fcw[ 0 ]), param_sizes_[ 10 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.fcb[ 0 ]), param_sizes_[ 11 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.fcprojw[ 0 ]), param_sizes_[ 12 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.fcprojb[ 0 ]), param_sizes_[ 13 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.lnfw[ 0 ]), param_sizes_[ 14 ] * sizeof( float ) );
			model_file.read( reinterpret_cast<char*>(&params_.lnfb[ 0 ]), param_sizes_[ 15 ] * sizeof( float ) );

			// TOD): Vectors or Tensors here
			m_memory_ = nullptr;
			v_memory_ = nullptr;
		}

		void backward() {
			// double check we forwarded previously, with targets
			if ( mean_loss_ == -1.0f ) {
				throw std::runtime_error( "Improper state: forward() not called with targets." );
			}

			// lazily allocate the Tensors for gradients of the weights and activations, if needed
			// TODO: Add model flag for training mode vs inference mode tp model configuration
			if ( grads_.wte.empty() ) {
				initialize_parameter_tensors( grads_ );
				initialize_activation_tensors( grads_acts_ );
			}

			// convenience shortcuts (and size_t to help prevent int overflow)
			size_t B = batch_size_;
			size_t T = seq_len_;
			size_t V = config_.vocab_size;
			size_t Vp = config_.padded_vocab_size;
			size_t L = config_.num_layers;
			size_t NH = config_.num_heads;
			size_t C = config_.channels;

			// backward pass: go in the reverse order of the forward pass, and call backward() functions
			// we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
			float dloss_mean = 1.0f / (B * T);
			for ( int i = 0; i < B * T; i++ ) {
				grads_acts_.losses[ i ] = dloss_mean;
			}

			//crossentropy_softmax_backward( grads_acts_.logits.data(), grads_acts_.losses.data(), acts_.probs.data(), targets_, B, T, V, Vp );
			//matmul_backward( grads_acts_.lnf.data(), grads_.wte.data(), NULL, grads_acts_.logits.data(), acts_.lnf.data(), params_.wte.data(), B, T, C, Vp );
			//float* residual = acts_.residual3.data() + (L - 1) * B * T * C; // last layer's residual
			//float* dresidual = grads_acts_.residual3.data() + (L - 1) * B * T * C; // write to last layer's residual
			//layernorm_backward( dresidual, grads_.lnfw.data(), grads_.lnfb.data(), grads_acts_.lnf.data(), residual, params_.lnfw.data(), acts_.lnf_mean.data(), acts_.lnf_rstd.data(), B, T, C );

			for ( int l = L - 1; l >= 0; l-- ) {
				//residual = l == 0 ? acts_.encoded.data() : acts_.residual3.data() + (l - 1) * B * T * C;
				//dresidual = l == 0 ? grads_acts_.encoded.data() : grads_acts_.residual3.data() + (l - 1) * B * T * C;

				// get the pointers of the weights for this layer
				float* l_ln1w = params_.ln1w.data() + l * C;
				float* l_qkvw = params_.qkvw.data() + l * 3 * C * C;
				float* l_attprojw = params_.attprojw.data() + l * C * C;
				float* l_ln2w = params_.ln2w.data() + l * C;
				float* l_fcw = params_.fcw.data() + l * 4 * C * C;
				float* l_fcprojw = params_.fcprojw.data() + l * C * 4 * C;

				// get the pointers of the gradients of the weights for this layer
				float* dl_ln1w = grads_.ln1w.data() + l * C;
				float* dl_ln1b = grads_.ln1b.data() + l * C;

				float* dl_qkvw = grads_.qkvw.data() + l * 3 * C * C;
				float* dl_qkvb = grads_.qkvb.data() + l * 3 * C;
				float* dl_attprojw = grads_.attprojw.data() + l * C * C;
				float* dl_attprojb = grads_.attprojb.data() + l * C;
				float* dl_ln2w = grads_.ln2w.data() + l * C;
				float* dl_ln2b = grads_.ln2b.data() + l * C;
				float* dl_fcw = grads_.fcw.data() + l * 4 * C * C;
				float* dl_fcb = grads_.fcb.data() + l * 4 * C;
				float* dl_fcprojw = grads_.fcprojw.data() + l * C * 4 * C;
				float* dl_fcprojb = grads_.fcprojb.data() + l * C;
				// get the pointers of the activations for this layer
				float* l_ln1 = acts_.ln1.data() + l * B * T * C;
				float* l_ln1_mean = acts_.ln1_mean.data() + l * B * T;
				float* l_ln1_rstd = acts_.ln1_rstd.data() + l * B * T;
				float* l_qkv = acts_.qkv.data() + l * B * T * 3 * C;
				float* l_atty = acts_.atty.data() + l * B * T * C;
				float* l_att = acts_.att.data() + l * B * NH * T * T;
				float* l_residual2 = acts_.residual2.data() + l * B * T * C;
				float* l_ln2 = acts_.ln2.data() + l * B * T * C;
				float* l_ln2_mean = acts_.ln2_mean.data() + l * B * T;
				float* l_ln2_rstd = acts_.ln2_rstd.data() + l * B * T;
				float* l_fch = acts_.fch.data() + l * B * T * 4 * C;
				float* l_fch_gelu = acts_.fch_gelu.data() + l * B * T * 4 * C;
				// get the pointers of the gradients of the activations for this layer
				float* dl_ln1 = grads_acts_.ln1.data() + l * B * T * C;
				float* dl_qkv = grads_acts_.qkv.data() + l * B * T * 3 * C;
				float* dl_atty = grads_acts_.atty.data() + l * B * T * C;
				float* dl_preatt = grads_acts_.preatt.data() + l * B * NH * T * T;
				float* dl_att = grads_acts_.att.data() + l * B * NH * T * T;
				float* dl_attproj = grads_acts_.attproj.data() + l * B * T * C;
				float* dl_residual2 = grads_acts_.residual2.data() + l * B * T * C;
				float* dl_ln2 = grads_acts_.ln2.data() + l * B * T * C;
				float* dl_fch = grads_acts_.fch.data() + l * B * T * 4 * C;
				float* dl_fch_gelu = grads_acts_.fch_gelu.data() + l * B * T * 4 * C;
				float* dl_fcproj = grads_acts_.fcproj.data() + l * B * T * C;
				float* dl_residual3 = grads_acts_.residual3.data() + l * B * T * C;

				// backprop this layer
				//residual_backward( dl_residual2, dl_fcproj, dl_residual3, B * T * C );
				//matmul_backward( dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C );
				//gelu_backward( dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C );
				//matmul_backward( dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C );
				//layernorm_backward( dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C );
				//residual_backward( dresidual, dl_attproj, dl_residual2, B * T * C );
				//matmul_backward( dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C );
				//attention_backward( dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH );
				//matmul_backward( dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C );
				//layernorm_backward( dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C );
			}
			//encoder_backward( grads_.wte.data(), grads_.wpe.data(), grads_acts_.encoded.data(), inputs_, B, T, C );
		}

		void forward( const Tensor<int>& inputs, const Tensor<int>& targets, size_t B, size_t T ) {
			// convenience parameters (size_t to help prevent int overflow)
			size_t V = config_.vocab_size;
			size_t Vp = config_.padded_vocab_size;
			size_t L = config_.num_layers;
			size_t NH = config_.num_heads;
			size_t C = config_.channels;

			// validate inputs, all indices must be in the range [0, V)
			// TJT: This seems unnecessary - review 
			for ( int i = 0; i < B * T; i++ ) {
				assert( 0 <= inputs[ i ] && inputs[ i ] < V );
				if ( !targets.empty() ) {
					assert( 0 <= targets[ i ] && targets[ i ] < V );
				}
			}

			// allocate space for all the activations if needed (done here, lazily)
			if ( acts_.encoded.empty() ) {
				// record the current B,T as well
				batch_size_ = B;
				seq_len_ = T;

				// initialize the activation Tensors
				initialize_activation_tensors( acts_ );

				// also create Tensors fpr caching inputs and targets
				inputs_.reshape( { B * T } );
				targets_.reshape( { B * T } ); // might be unused if we never have targets but it's small
			}
			else {
				// validate B,T is consistent with how we've allocated Tensors previously
				// in principle we could get more clever here in the future, for now this is safest
				if ( B != batch_size_ || T != seq_len_ ) {
					std::cerr << std::format( "Model: B={} T={}, Desired: B={} T={}\n", B, T, (int)batch_size_, (int)seq_len_ );
					exit( EXIT_FAILURE );
				}
			}

			// cache the inputs/targets
			// FIXME
			/*inputs_ = inputs;
			if ( !targets.empty() ) {
				targets_ = targets;
			}*/

			// forward pass
			float* residual;
			//encoder_forward( acts_.encoded.data(), inputs, params_.wte.data(), params_.wpe.data(), B, T, C ); // encoding goes into residual[0]

			for ( int l = 0; l < L; l++ ) {
				residual = l == 0 ? acts_.encoded.data() : acts_.residual3.data() + (l - 1) * B * T * C;

				// get the pointers of the weights and biases for this layer
				float* l_ln1w = params_.ln1w.data() + l * C;
				float* l_ln1b = params_.ln1b.data() + l * C;
				float* l_qkvw = params_.qkvw.data() + l * 3 * C * C;
				float* l_qkvb = params_.qkvb.data() + l * 3 * C;
				float* l_attprojw = params_.attprojw.data() + l * C * C;
				float* l_attprojb = params_.attprojb.data() + l * C;
				float* l_ln2w = params_.ln2w.data() + l * C;
				float* l_ln2b = params_.ln2b.data() + l * C;
				float* l_fcw = params_.fcw.data() + l * 4 * C * C;
				float* l_fcb = params_.fcb.data() + l * 4 * C;
				float* l_fcprojw = params_.fcprojw.data() + l * C * 4 * C;
				float* l_fcprojb = params_.fcprojb.data() + l * C;

				// get the pointers of the activations for this layer
				float* l_ln1 = acts_.ln1.data() + l * batch_size_ * seq_len_ * C;
				float* l_ln1_mean = acts_.ln1_mean.data() + l * batch_size_ * seq_len_;
				float* l_ln1_rstd = acts_.ln1_rstd.data() + l * batch_size_ * seq_len_;
				float* l_qkv = acts_.qkv.data() + l * batch_size_ * seq_len_ * 3 * C;
				float* l_atty = acts_.atty.data() + l * batch_size_ * seq_len_ * C;
				float* l_preatt = acts_.preatt.data() + l * batch_size_ * NH * seq_len_ * seq_len_;
				float* l_att = acts_.att.data() + l * batch_size_ * NH * seq_len_ * seq_len_;
				float* l_attproj = acts_.attproj.data() + l * batch_size_ * seq_len_ * C;
				float* l_residual2 = acts_.residual2.data() + l * batch_size_ * seq_len_ * C;
				float* l_ln2 = acts_.ln2.data() + l * batch_size_ * seq_len_ * C;
				float* l_ln2_mean = acts_.ln2_mean.data() + l * batch_size_ * seq_len_;
				float* l_ln2_rstd = acts_.ln2_rstd.data() + l * batch_size_ * seq_len_;
				float* l_fch = acts_.fch.data() + l * batch_size_ * seq_len_ * 4 * C;
				float* l_fch_gelu = acts_.fch_gelu.data() + l * batch_size_ * seq_len_ * 4 * C;
				float* l_fcproj = acts_.fcproj.data() + l * batch_size_ * seq_len_ * C;
				float* l_residual3 = acts_.residual3.data() + l * batch_size_ * seq_len_ * C;

				// now do the forward pass
				//layernorm_forward( l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, batch_size_, seq_len_, C );
				//matmul_forward( l_qkv, l_ln1, l_qkvw, l_qkvb, batch_size_, seq_len_, C, 3 * C );
				//attention_forward( l_atty, l_preatt, l_att, l_qkv, batch_size_, seq_len_, C, NH );
				//matmul_forward( l_attproj, l_atty, l_attprojw, l_attprojb, batch_size_, seq_len_, C, C );
				//residual_forward( l_residual2, residual, l_attproj, batch_size_ * seq_len_ * C );
				//layernorm_forward( l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, batch_size_, seq_len_, C );
				//matmul_forward( l_fch, l_ln2, l_fcw, l_fcb, batch_size_, seq_len_, C, 4 * C );
				//gelu_forward( l_fch_gelu, l_fch, batch_size_ * seq_len_ * 4 * C );
				//matmul_forward( l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, batch_size_, seq_len_, 4 * C, C );
				//residual_forward( l_residual3, l_residual2, l_fcproj, batch_size_ * seq_len_ * C );
			}
			residual = acts_.residual3.data() + (L - 1) * batch_size_ * seq_len_ * C; // last residual is in residual3
			//layernorm_forward( acts_.lnf.data(), acts_.lnf_mean.data(), acts_.lnf_rstd.data(), residual, params_.lnfw.data(), params_.lnfb.data(), batch_size_, seq_len_, C );
			//matmul_forward( acts_.logits.data(), acts_.lnf.data(), params_.wte.data(), NULL, batch_size_, seq_len_, C, Vp );
			//softmax_forward( acts_.probs.data(), acts_.logits.data(), batch_size_, seq_len_, V, Vp );

			// also forward the cross-entropy loss function if we have the targets
			if ( !targets.empty() ) {
				//crossentropy_forward( acts_.losses.data(), acts_.probs.data(), targets, batch_size_, seq_len_, Vp );
				// for convenience also evaluate the mean loss
				float mean_loss = 0.0f;
				for ( int i = 0; i < batch_size_ * seq_len_; i++ ) {
					mean_loss += acts_.losses[ i ];
				}
				mean_loss /= batch_size_ * seq_len_;
				mean_loss_ = mean_loss;
			}
			else {
				// if we don't have targets, we don't have a loss
				mean_loss_ = -1.0f;
			}
		}

		void zero_grads() {
			// Check required as grads are lazy loaded!
			if ( !grads_.wte.empty() ) {
				zero_param_grads();
			}
			if ( !grads_acts_.encoded.empty() ) {
				zero_activation_grads();
			}
		}

		void update( float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t ) {
			// reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

			// lazily allocate the memory for m_memory and v_memory
			//if (m_memory_.empty() ) {
			//	m_memory_ = (float*)calloc( model->num_parameters, sizeof( float ) );
			//	v_memory_ = (float*)calloc( model->num_parameters, sizeof( float ) );
			//}

			//for ( size_t i = 0; i < num_parameters_; i++ ) {
			//	float param = model->params_memory[ i ];
			//	float grad = model->grads_memory[ i ];

			//	// update the first moment (momentum)
			//	float m = beta1 * model->m_memory[ i ] + (1.0f - beta1) * grad;
			//	// update the second moment (RMSprop)
			//	float v = beta2 * model->v_memory[ i ] + (1.0f - beta2) * grad * grad;
			//	// bias-correct both moments
			//	float m_hat = m / (1.0f - powf( beta1, t ));
			//	float v_hat = v / (1.0f - powf( beta2, t ));

			//	// update
			//	model->m_memory[ i ] = m;
			//	model->v_memory[ i ] = v;
			//	model->params_memory[ i ] -= learning_rate * (m_hat / (sqrtf( v_hat ) + eps) + weight_decay * param);
			//}
		}

		void print() const {
			std::cout << "GPT-2 Model: " << std::endl;
			std::cout << "max_seq_len: " << config_.max_seq_len << std::endl;
			std::cout << "vocab_size: " << config_.vocab_size << std::endl;
			std::cout << "padded_vocab_size: " << config_.padded_vocab_size << std::endl;
			std::cout << "num_layers: " << config_.num_layers << std::endl;
			std::cout << "num_heads: " << config_.num_heads << std::endl;
			std::cout << "channels: " << config_.channels << std::endl;
			std::cout << "num_parameters: " << num_parameters_ << std::endl;
			std::cout << "batch_size_: " << batch_size_ << std::endl;
			std::cout << "seq_len_: " << seq_len_ << std::endl;
			std::cout << "is_training_: " << is_training_ << std::endl;
		}

	private:

		void initialize_parameter_tensor_sizes() {
			// Shorthand notation
			size_t Vp = config_.padded_vocab_size;
			size_t C = config_.channels;
			size_t maxT = config_.max_seq_len;
			size_t L = config_.num_layers;

			param_sizes_[ 0 ] = Vp * C; // wte
			param_sizes_[ 1 ] = maxT * C; // wpe

			param_sizes_[ 2 ] = L * C; // ln1w
			param_sizes_[ 3 ] = L * C; // ln1b
			param_sizes_[ 4 ] = L * (3 * C) * C; // qkvw
			param_sizes_[ 5 ] = L * (3 * C); // qkvb
			param_sizes_[ 6 ] = L * C * C; // attprojw
			param_sizes_[ 7 ] = L * C; // attprojb
			param_sizes_[ 8 ] = L * C; // ln2w
			param_sizes_[ 9 ] = L * C; // ln2b
			param_sizes_[ 10 ] = L * (4 * C) * C; // fcw
			param_sizes_[ 11 ] = L * (4 * C); // fcb
			param_sizes_[ 12 ] = L * C * (4 * C); // fcprojw
			param_sizes_[ 13 ] = L * C; // fcprojb

			param_sizes_[ 14 ] = C; // lnfw
			param_sizes_[ 15 ] = C; // lnfb

			num_parameters_ = 0;
			for ( size_t i = 0; i < NumberOfParameterTensors; i++ ) {
				num_parameters_ += param_sizes_[ i ];
			}
		}

		void initialize_parameter_tensors( ParameterTensors& params ) {
			// Model input encoding weights 
			params.wte = std::vector<float>( param_sizes_[ 0 ] );
			params.wpe = std::vector<float>( param_sizes_[ 1 ] );

			// Linear layer 1 weights and biases
			params.ln1w = std::vector<float>( param_sizes_[ 2 ] );
			params.ln1b = std::vector<float>( param_sizes_[ 3 ] );

			// Query, Key, Value layer weights and biases
			params.qkvw = std::vector<float>( param_sizes_[ 4 ] );
			params.qkvb = std::vector<float>( param_sizes_[ 5 ] );

			// Attention projection layer weights and biases
			params.attprojw = std::vector<float>( param_sizes_[ 6 ] );
			params.attprojb = std::vector<float>( param_sizes_[ 7 ] );

			// Layer normalization 2 weights and biases
			params.ln2w = std::vector<float>( param_sizes_[ 8 ] );
			params.ln2b = std::vector<float>( param_sizes_[ 9 ] );

			// Fully connected layer weights and biases
			params.fcw = std::vector<float>( param_sizes_[ 10 ] );
			params.fcb = std::vector<float>( param_sizes_[ 11 ] );

			// Fully connected projection layer weights and biases
			params.fcprojw = std::vector<float>( param_sizes_[ 12 ] );
			params.fcprojb = std::vector<float>( param_sizes_[ 13 ] );

			// Layer normalization final weights and biases
			params.lnfw = std::vector<float>( param_sizes_[ 14 ] );
			params.lnfb = std::vector<float>( param_sizes_[ 15 ] );
		}

		void initialize_activation_tensor_sizes() {
			// The batch size B and sequence length T for the current forward pass
			size_t B = batch_size_;
			size_t T = seq_len_;

			size_t C = config_.channels;
			size_t NH = config_.num_heads;
			size_t L = config_.num_layers;
			size_t Vp = config_.padded_vocab_size;

			// TODO: These should be activation shapes so that proper tensors can be created.

			act_sizes_[ 0 ] = B * T * C; // encoded
			act_sizes_[ 1 ] = L * B * T * C; // ln1
			act_sizes_[ 2 ] = L * B * T; // ln1_mean
			act_sizes_[ 3 ] = L * B * T; // ln1_rstd
			act_sizes_[ 4 ] = L * B * T * 3 * C; // qkv
			act_sizes_[ 5 ] = L * B * T * C; // atty
			act_sizes_[ 6 ] = L * B * NH * T * T; // preatt
			act_sizes_[ 7 ] = L * B * NH * T * T; // att
			act_sizes_[ 8 ] = L * B * T * C; // attproj
			act_sizes_[ 9 ] = L * B * T * C; // residual2
			act_sizes_[ 10 ] = L * B * T * C; // ln2
			act_sizes_[ 11 ] = L * B * T; // ln2_mean
			act_sizes_[ 12 ] = L * B * T; // ln2_rstd
			act_sizes_[ 13 ] = L * B * T * 4 * C; // fch
			act_sizes_[ 14 ] = L * B * T * 4 * C; // fch_gelu
			act_sizes_[ 15 ] = L * B * T * C; // fcproj
			act_sizes_[ 16 ] = L * B * T * C; // residual3
			act_sizes_[ 17 ] = B * T * C; // lnf
			act_sizes_[ 18 ] = B * T; // lnf_mean
			act_sizes_[ 19 ] = B * T; // lnf_rstd
			act_sizes_[ 20 ] = B * T * Vp; // logits
			act_sizes_[ 21 ] = B * T * Vp; // probs
			act_sizes_[ 22 ] = B * T; // losses

			num_activations_ = 0;
			for ( size_t i = 0; i < NumberOfActivationTensors; i++ ) {
				num_activations_ += act_sizes_[ i ];
			}
		}

		void initialize_activation_tensors( ActivationTensors& acts ) {

			// TJT: These are the flat vector tensor representations...
			// TODO: These should be activation shapes so that proper tensors can be created.

			acts.encoded = std::vector<float>( act_sizes_[ 0 ] );
			acts.ln1 = std::vector<float>( act_sizes_[ 1 ] );
			acts.ln1_mean = std::vector<float>( act_sizes_[ 2 ] );
			acts.ln1_rstd = std::vector<float>( act_sizes_[ 3 ] );
			acts.qkv = std::vector<float>( act_sizes_[ 4 ] );
			acts.atty = std::vector<float>( act_sizes_[ 5 ] );
			acts.preatt = std::vector<float>( act_sizes_[ 6 ] );
			acts.att = std::vector<float>( act_sizes_[ 7 ] );
			acts.attproj = std::vector<float>( act_sizes_[ 8 ] );
			acts.residual2 = std::vector<float>( act_sizes_[ 9 ] );
			acts.ln2 = std::vector<float>( act_sizes_[ 10 ] );
			acts.ln2_mean = std::vector<float>( act_sizes_[ 11 ] );
			acts.ln2_rstd = std::vector<float>( act_sizes_[ 12 ] );
			acts.fch = std::vector<float>( act_sizes_[ 13 ] );
			acts.fch_gelu = std::vector<float>( act_sizes_[ 14 ] );
			acts.fcproj = std::vector<float>( act_sizes_[ 15 ] );
			acts.residual3 = std::vector<float>( act_sizes_[ 16 ] );
			acts.lnf = std::vector<float>( act_sizes_[ 17 ] );
			acts.lnf_mean = std::vector<float>( act_sizes_[ 18 ] );
			acts.lnf_rstd = std::vector<float>( act_sizes_[ 19 ] );
			acts.logits = std::vector<float>( act_sizes_[ 20 ] );
			acts.probs = std::vector<float>( act_sizes_[ 21 ] );
			acts.losses = std::vector<float>( act_sizes_[ 22 ] );
		}

		void zero_param_grads()
		{
			std::fill( grads_.wte.begin(), grads_.wte.end(), 0);
			std::fill( grads_.wpe.begin(), grads_.wpe.end(), 0 );

			std::fill( grads_.ln1w.begin(), grads_.ln1w.end(), 0 );
			std::fill( grads_.ln1b.begin(), grads_.ln1b.end(), 0 );

			std::fill( grads_.qkvw.begin(), grads_.qkvw.end(), 0 );
			std::fill( grads_.qkvb.begin(), grads_.qkvb.end(), 0 );

			std::fill( grads_.attprojw.begin(), grads_.attprojw.end(), 0 );
			std::fill( grads_.attprojb.begin(), grads_.attprojb.end(), 0 );

			std::fill( grads_.ln2w.begin(), grads_.ln2w.end(), 0 );
			std::fill( grads_.ln2b.begin(), grads_.ln2b.end(), 0 );

			std::fill( grads_.fcw.begin(), grads_.fcw.end(), 0 );
			std::fill( grads_.fcb.begin(), grads_.fcb.end(), 0 );

			std::fill( grads_.fcprojw.begin(), grads_.fcprojw.end(), 0 );
			std::fill( grads_.fcprojb.begin(), grads_.fcprojb.end(), 0 );

			std::fill( grads_.lnfw.begin(), grads_.lnfw.end(), 0 );
			std::fill( grads_.lnfb.begin(), grads_.lnfb.end(), 0 );
		}

		void zero_activation_grads()
		{
			std::fill( grads_acts_.encoded.begin(), grads_acts_.encoded.end(), 0);
			std::fill( grads_acts_.ln1.begin(), grads_acts_.ln1.end(), 0 );
			std::fill( grads_acts_.ln1_mean.begin(), grads_acts_.ln1_mean.end(), 0 );
			std::fill( grads_acts_.ln1_rstd.begin(), grads_acts_.ln1_rstd.end(), 0 );
			std::fill( grads_acts_.qkv.begin(), grads_acts_.qkv.end(), 0 );
			std::fill( grads_acts_.atty.begin(), grads_acts_.atty.end(), 0 );
			std::fill( grads_acts_.preatt.begin(), grads_acts_.preatt.end(), 0 );
			std::fill( grads_acts_.att.begin(), grads_acts_.att.end(), 0 );
			std::fill( grads_acts_.attproj.begin(), grads_acts_.attproj.end(), 0 );
			std::fill( grads_acts_.residual2.begin(), grads_acts_.residual2.end(), 0 );
			std::fill( grads_acts_.ln2.begin(), grads_acts_.ln2.end(), 0 );
			std::fill( grads_acts_.ln2_mean.begin(), grads_acts_.ln2_mean.end(), 0 );
			std::fill( grads_acts_.ln2_rstd.begin(), grads_acts_.ln2_rstd.end(), 0 );
			std::fill( grads_acts_.fch.begin(), grads_acts_.fch.end(), 0 );
			std::fill( grads_acts_.fch_gelu.begin(), grads_acts_.fch_gelu.end(), 0 );
			std::fill( grads_acts_.fcproj.begin(), grads_acts_.fcproj.end(), 0 );
			std::fill( grads_acts_.residual3.begin(), grads_acts_.residual3.end(), 0 );
			std::fill( grads_acts_.lnf.begin(), grads_acts_.lnf.end(), 0 );
			std::fill( grads_acts_.lnf_mean.begin(), grads_acts_.lnf_mean.end(), 0 );
			std::fill( grads_acts_.lnf_rstd.begin(), grads_acts_.lnf_rstd.end(), 0 );
			std::fill( grads_acts_.logits.begin(), grads_acts_.logits.end(), 0 );
			std::fill( grads_acts_.probs.begin(), grads_acts_.probs.end(), 0 );
			std::fill( grads_acts_.losses.begin(), grads_acts_.losses.end(), 0 );
		}

		ModelConfig config_;

		// the weights (parameters) of the model, and their sizes
		ParameterTensors params_;
		size_t param_sizes_[ NumberOfParameterTensors ];
		size_t num_parameters_;

		// gradients of the weights
		ParameterTensors grads_;

		// buffers for the AdamW optimizer
		// TODO: Vector or Tensor here
		float* m_memory_{ nullptr };
		float* v_memory_{ nullptr };

		// the activations of the model, and their sizes
		ActivationTensors acts_;
		size_t act_sizes_[ NumberOfActivationTensors ];
		size_t num_activations_;

		// gradients of the activations
		ActivationTensors grads_acts_;

		// other run state configuration
		int batch_size_{ 0 }; // the batch size (B) of current forward pass
		int seq_len_{ 0 }; // the sequence length (T) of current forward pass

		bool is_training_{ true };

		Tensor<int> inputs_; // the input tokens for the current forward pass
		Tensor<int> targets_; // the target tokens for the current forward pass

		float mean_loss_{ -1.0f }; // after a forward pass with targets, will be populated with the mean loss
	};
}