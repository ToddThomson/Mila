/**
 * @file Gpt2.CheckpointReader.ixx
 * @brief Read GPT-2 parameter blobs from a binary checkpoint file.
 *
 * Provides helpers to compute parameter sizes, allocate host parameter vectors,
 * and load parameter blobs from the checkpoint header/stream.
 */

module;
#include <array>
#include <cstdint>
#include <fstream>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

export module Gpt2.CheckpointReader;

import Gpt2.Config;
import Mila;

namespace Mila::Gpt2
{
    constexpr int Gpt2ModelHeaderSize = 256;
    constexpr int NumberOfParameterTensors = 16;

    /**
     * @brief Flat host-side parameter storage for checkpoint blobs.
     *
     * Each field corresponds to a contiguous FP32 blob in the checkpoint file.
     * The reader allocates these vectors (by size) and fills them from file.
     */
    export struct ParameterTensors
    {
        std::vector<float> wte;
        std::vector<float> wpe;

        std::vector<float> ln1w;
        std::vector<float> ln1b;

        std::vector<float> qkvw;
        std::vector<float> qkvb;

        std::vector<float> attprojw;
        std::vector<float> attprojb;

        std::vector<float> ln2w;
        std::vector<float> ln2b;

        std::vector<float> fcw;
        std::vector<float> fcb;

        std::vector<float> fcprojw;
        std::vector<float> fcprojb;

        std::vector<float> lnfw;
        std::vector<float> lnfb;
    };

    /**
     * @brief Compute sizes of parameter blobs given model config.
     *
     * @param cfg ModelConfig (max_seq_len, padded_vocab_size, num_layers, num_heads, channels)
     * @param out_param_sizes Output array of length NumberOfParameterTensors populated with element counts
     * @param out_num_parameters Optional output total parameter count (sum of sizes)
     */
    export void computeParameterSizes(
        const Gpt2Config& cfg,
        std::array<size_t, NumberOfParameterTensors>& out_param_sizes,
        size_t* out_num_parameters = nullptr )
    {
        size_t Vp = static_cast<size_t>(cfg.padded_vocab_size);
        size_t C = static_cast<size_t>(cfg.channels);
        size_t maxT = static_cast<size_t>(cfg.max_seq_len);
        size_t L = static_cast<size_t>(cfg.num_layers);

        out_param_sizes[ 0 ] = Vp * C;                     // wte
        out_param_sizes[ 1 ] = maxT * C;                   // wpe

        out_param_sizes[ 2 ] = L * C;                      // ln1w
        out_param_sizes[ 3 ] = L * C;                      // ln1b

        out_param_sizes[ 4 ] = L * (3 * C) * C;            // qkvw
        out_param_sizes[ 5 ] = L * (3 * C);                // qkvb

        out_param_sizes[ 6 ] = L * C * C;                  // attprojw
        out_param_sizes[ 7 ] = L * C;                      // attprojb

        out_param_sizes[ 8 ] = L * C;                      // ln2w
        out_param_sizes[ 9 ] = L * C;                      // ln2b

        out_param_sizes[ 10 ] = L * (4 * C) * C;           // fcw
        out_param_sizes[ 11 ] = L * (4 * C);               // fcb

        out_param_sizes[ 12 ] = L * C * (4 * C);           // fcprojw
        out_param_sizes[ 13 ] = L * C;                     // fcprojb

        out_param_sizes[ 14 ] = C;                         // lnfw
        out_param_sizes[ 15 ] = C;                         // lnfb

        if ( out_num_parameters )
        {
            size_t total = 0;
            for ( size_t i = 0; i < NumberOfParameterTensors; ++i )
                total += out_param_sizes[ i ];

            *out_num_parameters = total;
        }
    }

    /**
     * @brief Allocate host-side ParameterTensors vectors according to param sizes.
     *
     * @param sizes Array of parameter sizes computed by computeParameterSizes()
     * @param out_params ParameterTensors that will be allocated
     */
    export void allocateParameterTensors(
        const std::array<size_t, NumberOfParameterTensors>& sizes,
        ParameterTensors& out_params )
    {
        out_params.wte = std::vector<float>( sizes[ 0 ] );
        out_params.wpe = std::vector<float>( sizes[ 1 ] );

        out_params.ln1w = std::vector<float>( sizes[ 2 ] );
        out_params.ln1b = std::vector<float>( sizes[ 3 ] );

        out_params.qkvw = std::vector<float>( sizes[ 4 ] );
        out_params.qkvb = std::vector<float>( sizes[ 5 ] );

        out_params.attprojw = std::vector<float>( sizes[ 6 ] );
        out_params.attprojb = std::vector<float>( sizes[ 7 ] );

        out_params.ln2w = std::vector<float>( sizes[ 8 ] );
        out_params.ln2b = std::vector<float>( sizes[ 9 ] );

        out_params.fcw = std::vector<float>( sizes[ 10 ] );
        out_params.fcb = std::vector<float>( sizes[ 11 ] );

        out_params.fcprojw = std::vector<float>( sizes[ 12 ] );
        out_params.fcprojb = std::vector<float>( sizes[ 13 ] );

        out_params.lnfw = std::vector<float>( sizes[ 14 ] );
        out_params.lnfb = std::vector<float>( sizes[ 15 ] );
    }

    /**
     * @brief Read checkpoint file into host-side ParameterTensors and fill param_sizes.
     *
     * The ModelConfig reference is updated from the checkpoint header (hyperparameters)
     * so callers can use the updated config for subsequent graph building.
     *
     * Throws std::runtime_error on I/O or format errors.
     *
     * @param checkpoint_path Path to binary checkpoint
     * @param config IN/OUT: config will be overwritten with header values (max_seq_len, vocab_size, num_layers, num_heads, channels, padded_vocab_size)
     * @param out_params ParameterTensors that will be allocated and filled
     * @param out_param_sizes Filled with element counts for each parameter tensor
     */
    export void readCheckpoint(
        const std::string& checkpoint_path,
        Gpt2Config& config,
        ParameterTensors& out_params,
        std::array<size_t, NumberOfParameterTensors>& out_param_sizes )
    {
        std::ifstream model_file( checkpoint_path, std::ifstream::binary );
        if ( !model_file.is_open() )
        {
            throw std::runtime_error( std::format( "Could not open model file: {}", checkpoint_path ) );
        }

        // Read header
        std::array<int, Gpt2ModelHeaderSize> model_header{};
        model_file.read( reinterpret_cast<char*>(model_header.data()), Gpt2ModelHeaderSize * sizeof( int ) );

        if ( model_file.fail() )
        {
            throw std::runtime_error( std::format( "Failed to read header from {}", checkpoint_path ) );
        }

        // Validate magic & version (kept from original code)
        if ( model_header[ 0 ] != 20240326 )
        {
            throw std::runtime_error( std::format( "Invalid magic number for model file: {}", checkpoint_path ) );
        }

        if ( model_header[ 1 ] != 3 )
        {
            throw std::runtime_error( std::format( "Invalid version for model file: {}", checkpoint_path ) );
        }

        // Extract hyperparameters from header and update config
        size_t maxT, V, Vp, L, NH, C;
        config.max_seq_len = maxT = static_cast<size_t>(model_header[ 2 ]);
        config.vocab_size = V = static_cast<size_t>(model_header[ 3 ]);
        config.num_layers = L = static_cast<size_t>(model_header[ 4 ]);
        config.num_heads = NH = static_cast<size_t>(model_header[ 5 ]);
        config.channels = C = static_cast<size_t>(model_header[ 6 ]);
        config.padded_vocab_size = Vp = static_cast<size_t>(model_header[ 7 ]);

        // Compute param sizes and allocate host vectors
        computeParameterSizes( config, out_param_sizes );
        allocateParameterTensors( out_param_sizes, out_params );

        // Read blobs in the same order used by the original reader
        // Guard reads with file.fail checks
        if ( out_param_sizes[ 0 ] )
            model_file.read( reinterpret_cast<char*>(out_params.wte.data()), out_param_sizes[ 0 ] * sizeof( float ) );

        if ( out_param_sizes[ 1 ] )
            model_file.read( reinterpret_cast<char*>(out_params.wpe.data()), out_param_sizes[ 1 ] * sizeof( float ) );

        if ( out_param_sizes[ 2 ] )
            model_file.read( reinterpret_cast<char*>(out_params.ln1w.data()), out_param_sizes[ 2 ] * sizeof( float ) );

        if ( out_param_sizes[ 3 ] )
            model_file.read( reinterpret_cast<char*>(out_params.ln1b.data()), out_param_sizes[ 3 ] * sizeof( float ) );

        if ( out_param_sizes[ 4 ] )
            model_file.read( reinterpret_cast<char*>(out_params.qkvw.data()), out_param_sizes[ 4 ] * sizeof( float ) );

        if ( out_param_sizes[ 5 ] )
            model_file.read( reinterpret_cast<char*>(out_params.qkvb.data()), out_param_sizes[ 5 ] * sizeof( float ) );

        if ( out_param_sizes[ 6 ] )
            model_file.read( reinterpret_cast<char*>(out_params.attprojw.data()), out_param_sizes[ 6 ] * sizeof( float ) );

        if ( out_param_sizes[ 7 ] )
            model_file.read( reinterpret_cast<char*>(out_params.attprojb.data()), out_param_sizes[ 7 ] * sizeof( float ) );

        if ( out_param_sizes[ 8 ] )
            model_file.read( reinterpret_cast<char*>(out_params.ln2w.data()), out_param_sizes[ 8 ] * sizeof( float ) );

        if ( out_param_sizes[ 9 ] )
            model_file.read( reinterpret_cast<char*>(out_params.ln2b.data()), out_param_sizes[ 9 ] * sizeof( float ) );

        if ( out_param_sizes[ 10 ] )
            model_file.read( reinterpret_cast<char*>(out_params.fcw.data()), out_param_sizes[ 10 ] * sizeof( float ) );

        if ( out_param_sizes[ 11 ] )
            model_file.read( reinterpret_cast<char*>(out_params.fcb.data()), out_param_sizes[ 11 ] * sizeof( float ) );

        if ( out_param_sizes[ 12 ] )
            model_file.read( reinterpret_cast<char*>(out_params.fcprojw.data()), out_param_sizes[ 12 ] * sizeof( float ) );

        if ( out_param_sizes[ 13 ] )
            model_file.read( reinterpret_cast<char*>(out_params.fcprojb.data()), out_param_sizes[ 13 ] * sizeof( float ) );

        if ( out_param_sizes[ 14 ] )
            model_file.read( reinterpret_cast<char*>(out_params.lnfw.data()), out_param_sizes[ 14 ] * sizeof( float ) );

        if ( out_param_sizes[ 15 ] )
            model_file.read( reinterpret_cast<char*>(out_params.lnfb.data()), out_param_sizes[ 15 ] * sizeof( float ) );

        if ( model_file.fail() )
        {
            throw std::runtime_error( std::format( "Failed to read parameter blobs from {}", checkpoint_path ) );
        }
    }
}