// ----------------------------------------------------------------------------
// GPT2 training application
#include <time.h>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <variant>

import Mila;
import Utils.StepLogger;

import Gpt2App.Gpt2DataLoader;
import Gpt2App.Gpt2Tokenizer;
import Gpt2App.Gpt2Config;
import Gpt2App.Gpt2Model;

unsigned int random_u32( uint64_t* state ) {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	*state ^= *state >> 12;
	*state ^= *state << 25;
	*state ^= *state >> 27;
	return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32( uint64_t* state ) { // random float32 in [0,1)
	return (random_u32( state ) >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// CLI, poor man's argparse

void error_usage() {
	std::cerr << ("Usage:   ./train_gpt2fp32cu [options]\n");
	std::cerr << ("Options:\n");
	std::cerr << ("  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
	std::cerr << ("  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
	std::cerr << ("  -o <string> output log file (default = NULL)\n");
	std::cerr << ("  -b <int>    batch size B (default = 4)\n");
	std::cerr << ("  -t <int>    sequence length T (default = 1024)\n");
	std::cerr << ("  -l <float>  learning rate (default = 3e-4f)\n");
	std::cerr << ("  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
	std::cerr << ("  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
	std::cerr << ("  -s <int>    sample_every, how often we inference the model (default = 20)\n");
	std::cerr << ("  -g <int>    genT, how many steps of inference we do (default = 64)\n");
	std::cerr << ("  -c <string> compute type (default = auto)\n");
	std::cerr << ("  -h          print this help message\n");

	exit( EXIT_FAILURE );
}

int main( int argc, char* argv[] ) {
	using namespace Gpt2App;
	using namespace Mila::Dnn;

	Mila::Initialize();
	std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

	// read in the (optional) command line arguments
	std::string train_data_pattern = "data/datasets/tinyshakespeare/tiny_shakespeare_train.bin";
	std::string val_data_pattern = "data/datasets/tinyshakespeare/tiny_shakespeare_val.bin";
	std::string output_log_file;
	size_t B = 4; // batch size
	size_t T = 64; // sequence length max
	float learning_rate = 3e-4f;
	int val_loss_every = 20; // every how many steps do we eval validation loss?
	int val_max_steps = 20; // how many batches max do we eval for validation loss?
	int sample_every = 20; // every how many steps to do inference?
	int genT = 64; // number of steps of inference we will do
	std::string compute_type = "CUDA";
	
	for ( int i = 1; i < argc; i += 2 ) {
		if ( i + 1 >= argc ) { error_usage(); } // must have arg after flag
		if ( argv[ i ][ 0 ] != '-' ) { error_usage(); } // must start with dash
		if ( strlen( argv[ i ] ) != 2 ) { error_usage(); } // must be -x (one dash, one letter)
		// read in the args
		if ( argv[ i ][ 1 ] == 'i' ) { train_data_pattern = argv[ i + 1 ]; }
		else if ( argv[ i ][ 1 ] == 'j' ) { val_data_pattern = argv[ i + 1 ]; }
		else if ( argv[ i ][ 1 ] == 'o' ) { output_log_file = argv[ i + 1 ]; }
		else if ( argv[ i ][ 1 ] == 'b' ) { B = atoi( argv[ i + 1 ] ); }
		else if ( argv[ i ][ 1 ] == 't' ) { T = atoi( argv[ i + 1 ] ); }
		else if ( argv[ i ][ 1 ] == 'l' ) { learning_rate = atof( argv[ i + 1 ] ); }
		else if ( argv[ i ][ 1 ] == 'v' ) { val_loss_every = atoi( argv[ i + 1 ] ); }
		else if ( argv[ i ][ 1 ] == 'm' ) { val_max_steps = atoi( argv[ i + 1 ] ); }
		else if ( argv[ i ][ 1 ] == 's' ) { sample_every = atoi( argv[ i + 1 ] ); }
		else if ( argv[ i ][ 1 ] == 'g' ) { genT = atoi( argv[ i + 1 ] ); }
		else if ( argv[ i ][ 1 ] == 'c' ) { compute_type = argv[ i + 1 ]; }
		else if ( argv[ i ][ 1 ] == 'h' ) { error_usage(); }
		else { error_usage(); }
	}

	std::cout << "+-----------------------+---------------------------------------------------------+\n";
	std::cout << "| Parameter             | Value                                                   |\n";
	std::cout << "+-----------------------+---------------------------------------------------------+\n";
	std::cout << std::format( "| train data pattern    | {:<55} |\n", train_data_pattern );
	std::cout << std::format( "| val data pattern      | {:<55} |\n", val_data_pattern );
	std::cout << std::format( "| output log file       | {:<55} |\n", output_log_file.empty() ? "Logging to stdout" : output_log_file );
	std::cout << std::format( "| batch size B          | {:<55d} |\n", B );
	std::cout << std::format( "| sequence length T     | {:<55d} |\n", T );
	std::cout << std::format( "| learning rate         | {:<55f} |\n", learning_rate );
	std::cout << std::format( "| val_loss_every        | {:<55d} |\n", val_loss_every );
	std::cout << std::format( "| val_max_steps         | {:<55d} |\n", val_max_steps );
	std::cout << std::format( "| sample_every          | {:<55d} |\n", sample_every );
	std::cout << std::format( "| genT                  | {:<55d} |\n", genT );
	std::cout << std::format( "| compute type          | {:<55} |\n", compute_type );
	std::cout << "+-----------------------+---------------------------------------------------------+\n";

	// set up the Logger
	Mila::Utils::StepLogger logger( output_log_file );

	auto devices = Compute::list_devices();
	std::cout << "Available compute devices: ";
	for ( const auto& device : devices ) {
		std::cout << device << " ";
	}
	std::cout << std::endl;

	Compute::DeviceType device_type = Compute::toDeviceType( compute_type );

	// Set device context once before creating model
	//Compute::DeviceContext::instance().setDevice( compute_type );

	// Create model - it will use whatever device is currently set
	ModelConfig config;
	//auto model = std::make_unique<Gpt2Model<float>>( config, B, T );
	//model->fromCheckpoint("data/models/gpt2/gpt2_124M.bin");
	//model->print();

	// Use the model MR for tensor types
	//using MR = typename decltype(model)::element_type::MR;

	// build DataLoaders for both train and val
	auto train_loader = Gpt2DataLoader<int>( train_data_pattern, B, T, 0, 1, true );
	auto val_loader = Gpt2DataLoader<int>( val_data_pattern, B, T, 0, 1, false );

	int train_num_batches = train_loader.num_tokens() / (B * T); // let's do 1 epoch by default for now
	int val_num_batches = val_loader.num_tokens() / (B * T);
	if ( val_num_batches > val_max_steps ) {
		val_num_batches = val_max_steps;
	}
	std::cout << std::format( "val_num_batches: {:<55d} |\n", val_num_batches );

	// TJT: Adjusted
	val_num_batches = 5;

	// build the Tokenizer
	Tokenizer tokenizer = Tokenizer( "data/models/gpt2/gpt2_tokenizer.bin" );

	// Cpu Tensor for generating samples from the model.
	uint64_t rng_state = 1337;
	Tensor<int, MR> gen_tokens( std::vector<size_t>( { B, T } ) );

	// Training loop
	for ( int step = 0; step <= 40; step++ ) {
		// once in a while estimate the validation loss
		// TJT: Adjusted to never run
		if ( step % 10 == 100 ) {
			float val_loss = 0.0f;
			val_loader.reset();
			std::cout << "Calculating validation loss: .";

			for ( int i = 0; i < val_num_batches; i++ ) {
				val_loader.next_batch();
				// FIXME:
				//model->forward( val_loader.inputs(), val_loader.targets() );
				//val_loss += model->get_mean_loss();
				std::cout << ".";
			}
			std::cout << std::endl;
			val_loss /= val_num_batches;
			std::cout << "val loss: " << val_loss << std::endl;
		}

		// once in a while do model inference to print generated text
		if ( step > 0 && step % 2 /* 0 */ == 0 ) {
			// fill up gen_tokens with the GPT2_EOT, which kicks off the generation
			for ( int b = 0; b < B; ++b ) {
				for ( int t = 0; t < T; ++t ) {
					gen_tokens[ b, t ] = tokenizer.get_eot_token();
				}
			}
			// now sample from the model autoregressively
			std::cout << "generating:\n---\n";
			for ( int t = 1; t < genT; t++ ) {
				// note that inference is very wasteful here because for each token
				// we re-calculate the forward pass for all of (B,T) positions from scratch
				// but the inference here is just for sanity checking anyway
				// and we can maybe optimize a bit more later, with careful tests
				Tensor<int> empty_targets;
				//model->forward( gen_tokens, empty_targets );

				// furthermore, below we're only using b=0 (i.e. the first row) of all B rows
				// we're in principle running B "inference streams" in parallel here
				// but only using position 0
				// get the Vp-dimensional vector probs[0, t-1, :]
				//auto probs = model->getProbabilities();

				//float* probs = reinterpret_cast<float*>(acts.probs.data()) + ((t - 1) * model.get_config().padded_vocab_size);
				float coin = random_f32( &rng_state );
				// note we're only sampling from the first V elements, ignoring padding
				// (the probabilities in the padded region should be zero anyway)
				//int next_token = model->sampleMult( probs, model->get_config().vocab_size, t, coin );
				//gen_tokens[ 0, t ] = next_token;

				// print the generated token, either using the Tokenizer or a fallback
				//if ( tokenizer.init_ok ) {
				//const char* token_str = tokenizer.decode( next_token );
				//std::cout << token_str;
				//}
				//else {
				//    // fall back to printing the token id
				//    std::cout <<( "%d ", next_token );
				//}
				fflush( stdout );
			}
			std::cout << "\n---\n";
		}

		// Training step
		auto start_time = std::chrono::steady_clock::now();

		train_loader.next_batch();

		//model.forward( train_loader.inputs(), train_loader.targets() );
		//model.zero_grads();
		//model.backward();
		//model.update( 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1 );

		auto end_time = std::chrono::steady_clock::now();
		double time_elapsed_s = std::chrono::duration<double>( end_time - start_time ).count();
		//total_sum_iteration_time_s += time_elapsed_s;
		auto log_msg = std::format( "step {}/{}: train loss {} ({} ms)\n", step + 1, train_num_batches, model->get_mean_loss(), time_elapsed_s * 1000 );
		logger.log_step( step + 1, log_msg );
	}
	return 0;
}