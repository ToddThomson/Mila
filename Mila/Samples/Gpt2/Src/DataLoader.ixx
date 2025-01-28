/*
 * Copyright 2024..2025 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the Mila end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

module;
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/detail/raw_pointer_cast.h>

#include "../../../Src/Core/rand.h"

export module Gpt2.DataLoader;

import Dnn.Tensor;
import Helpers.Memory;
import Misc.Glob;

namespace Mila::Dnn::Gpt2
{
	//using Tensor = Mila::Dnn::Tensor;

    /// @brief Size of the GPT-2 token file header.
    constexpr int Gpt2TokenFileHeaderSize = 256;

    /// @brief Magic number for the GPT-2 tokenizer.
    constexpr int Gpt2TokenFileMagicNumber = 20240520;
	
    /**
    * @class DataLoader
    * @brief Class for loading and managing data for distributed training.
    */
    export class DataLoader {

    public:

        /**
        * @brief Constructor for DataLoader.
        * @param filename_pattern Pattern for filenames to load.
        * @param batch_size Size of each batch.
        * @param token_size Size of each token.
        * @param process_rank Rank of the current process.
        * @param num_processes Total number of processes.
        * @param should_shuffle Flag indicating whether to shuffle data.
        */
        DataLoader( const std::string& filename_pattern, size_t batch_size, size_t token_size, int process_rank, int num_processes, bool should_shuffle )
            : batch_size_( batch_size ), token_size_( token_size ), process_rank_( process_rank ), num_processes_( num_processes ), should_shuffle_( should_shuffle ) {
            // Initialize the data loader
            init( filename_pattern );
        }

        ~DataLoader() {
            free();
        }

        int num_tokens() const { 
            return num_tokens_;
        }

        Tensor<int>& inputs() {
            return inputs_;
        }
        
        const Tensor<int>& inputs() const {
            return inputs_; 
        }

        Tensor<int>& targets() {
            return targets_;
        }

        const Tensor<int>& targets() const { 
            return targets_;
        }

        /**
        * @brief Reset the data loader to the initial state.
        */
        void reset() {
            current_shard_idx_ = 0;
            current_sample_idx_ = 0;

            if ( should_shuffle_ ) {
                random_permutation( shard_indices_.data(), (int)glob_result_.gl_pathc, &shuffle_rng_);
            }

            load_shard_( (int)current_shard_idx_ );

            if ( should_shuffle_ ) {
                prepare_intra_shard_indices_();
            }
        }

        /**
        * @brief Load a batch of data.
        */
        void load_batch() {
            assert( !should_shuffle_ || (should_shuffle_ && !intra_shard_indices_.empty() ) );
            assert( current_sample_idx_ < shard_num_samples_ );

            size_t idx = should_shuffle_ ? intra_shard_indices_[ current_sample_idx_ ] : current_sample_idx_;
            size_t global_batch_offset_bytes = idx * total_batch_size_bytes_;
            int64_t current_offset = header_bytes_ + global_batch_offset_bytes + local_batch_offset_bytes_;

            size_t B = batch_size_;
            size_t T = token_size_;
            
            // read B*T+1 uint16_t tokens from the file into buffer
            tokens_file_.seekg( (int)current_offset );
            tokens_file_.read( reinterpret_cast<char*>( buffer_.data() ), ( B * T + 1 ) * sizeof(uint16_t) );
            
            // Decode the buffer into inputs and targets tensors
            for ( int i = 0; i < B * T; i++ ) {
                inputs_[ i ] = (int)buffer_[ i ];
                targets_[ i ] = (int)buffer_[ i + 1 ];
            }
        }

        /**
        * @brief Load the next batch of data.
        */
        void next_batch() {
            // if the next batch would go past the end of the file, advance the loader
            if ( current_sample_idx_ >= shard_num_samples_ ) {
                advance_();
            }
            load_batch();
            current_sample_idx_ += 1;
        }

        /**
        * @brief Resume loading from a specific shard and sample index.
        * @param current_shard_idx Index of the shard to resume from.
        * @param current_sample_idx Index of the sample to resume from.
        */
        void resume( size_t current_shard_idx, size_t current_sample_idx ) {
            // used during model resumption (-y 1) flag
            current_shard_idx_ = current_shard_idx;
            current_sample_idx_ = current_sample_idx;
            load_shard_( (int)current_shard_idx_ );
        }

        /**
        * @brief Print the current state of the DataLoader.
        */
        void print() {
            std::cout << "DataLoader: " << std::endl;
            std::cout << "  process_rank: " << process_rank_ << std::endl;
            std::cout << "  num_processes: " << num_processes_ << std::endl;
            std::cout << "  batch_size B: " << batch_size_ << std::endl;
            std::cout << "  token_siae T: " << token_size_ << std::endl;
            std::cout << "  num_tokens: " << num_tokens_ << std::endl;
            std::cout << "  train dataset num_batches: " << num_tokens_ / (batch_size_ * token_size_) << std::endl;
            std::cout << "  current_shard_idx: " << current_shard_idx_ << std::endl;
            std::cout << "  current_sample_idx: " << current_sample_idx_ << std::endl;
            std::cout << "  shard_num_samples: " << shard_num_samples_ << std::endl;
            std::cout << "  file_size_bytes: " << file_size_bytes_ << std::endl;
            std::cout << "  local_batch_offset_bytes: " << local_batch_offset_bytes_ << std::endl;
        }

        /**
        * @brief Free allocated resources.
        */
        void free() {
			// TODO: Review after glob implementation changes
            Mila::Misc::globfree( &glob_result_ );
        }

    private:
        /**
        * @brief Initialize the DataLoader.
        * @param filename_pattern Pattern for filenames to load.
        */
        void init( const std::string &filename_pattern ) {
            header_bytes_ = Gpt2TokenFileHeaderSize * sizeof( int );
            total_batch_size_bytes_ = ((num_processes_ * (batch_size_ * token_size_)) * sizeof( uint16_t ));
            local_batch_offset_bytes_ = process_rank_ * batch_size_ * token_size_ * sizeof( uint16_t );

            // glob to get the list of files matching the pattern, these are our data shards
            int glob_status = glob( filename_pattern.c_str(), 0, NULL, &glob_result_);
            
            if ( glob_status != 0 ) {
                throw std::runtime_error( std::format( "Error: failed to glob pattern: {}\n", filename_pattern ) );
            }
            if ( glob_result_.gl_pathc == 0 ) {
                throw std::runtime_error( std::format( "Error: no files found matching the pattern: {}\n", filename_pattern ) );
            }

            if ( should_shuffle_ ) {
                mt19937_state shuffle_rng;
                manual_seed( &shuffle_rng, 42 + process_rank_ );
                shuffle_rng_ = shuffle_rng;
                shard_indices_ = std::vector<int>( glob_result_.gl_pathc );
                init_identity_permutation( shard_indices_.data(), (int)glob_result_.gl_pathc);
                intra_shard_indices_.clear(); // = NULL;  // dynamically allocated allowing different shard sizes
            }

            // inspect and validate all shards so we don't get any runtime errors later
            // if too slow / too many shards, may wish to revisit later
            int64_t ntok_total = 0;
            for ( int shard_index = 0; shard_index < glob_result_.gl_pathc; shard_index++ ) {
                int64_t shard_ntok = load_shard_( shard_index );
                // we need at least one batch/shard, the way things are written right now.
                // can be relaxed a lot later.
                assert( shard_ntok >= (int64_t)(num_processes_ * batch_size_ * token_size_ + 1) );
                ntok_total += shard_ntok;
            }
            
            // debugging prints
            // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
            // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, glob_result_.gl_pathc);

            // allocate all the space we'll need
            buffer_.resize( (batch_size_ * token_size_ + 1));
            inputs_.reshape( { batch_size_ * token_size_ } );
            targets_.reshape( { batch_size_ * token_size_ } );
            
            num_tokens_ = ntok_total;

            // reset the loader, to initialize it
            reset();
        }

        /**
        * @brief Load a specific shard of data.
        * @param shard_index Index of the shard to load.
        * @return Number of tokens in the shard.
        */
        int64_t load_shard_( int shard_index ) {
            if ( should_shuffle_ ) {
                shard_index = shard_indices_[ shard_index ];
            }
            // use the first glob match as the filename for now
            const char* filename = glob_result_.gl_pathv[ shard_index ];
            
            // open the input file for reading. also only a single file can be opened at a time
            // TJT: This doesn't smell right - review
            if ( tokens_file_.is_open() ) {
                tokens_file_.close();
            }
           
            tokens_file_ = std::ifstream( filename, std::ios::in | std::ifstream::binary );
            if ( !tokens_file_.is_open() ) {
                throw std::runtime_error( std::format( "Failed to open token file: {}", filename ) );
            }
            
            // validate the header
            std::array<int, Gpt2TokenFileHeaderSize> header;
            tokens_file_.read( reinterpret_cast<char*>(header.data()), Gpt2TokenFileHeaderSize * sizeof( int ) );
            
            if ( header[ 0 ] != Gpt2TokenFileMagicNumber ) {
                throw std::runtime_error( std::format( "Invalid magic number in file: {}", filename) );
            }
            if ( header[ 1 ] != 1 ) {
                throw std::runtime_error( std::format( "Invalid version in file: {}", filename ) );
            }
            
            // number of tokens in the file
            // we expect some tokens in the file. this should never trip, right?
            int64_t ntok = header[ 2 ]; 
            assert( ntok > 0 );
            
            // determine the file size and make sure it is consistent with the number of tokens
            tokens_file_.seekg( 0, std::ios::end ); // seek to end of file
            file_size_bytes_ = tokens_file_.tellg(); // read the offset, i.e. file size
            tokens_file_.seekg( 0, std::ios::beg ); // seek back to the beginning
            
            // we expect ntok in the file to be consistent with filesize, assert that is the case
            int64_t expected_file_size = (header.size() * sizeof( int )) + (ntok * sizeof(uint16_t));
            if ( file_size_bytes_ != expected_file_size ) {
                throw std::runtime_error( std::format( "Unexpected file size. Expected: {} Actual: {} in file: {}", expected_file_size, file_size_bytes_, filename ) );
            }
            // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
            shard_num_samples_ = (ntok * sizeof( uint16_t ) - sizeof( uint16_t )) / total_batch_size_bytes_;
            
            return ntok;
        }

        /**
        * @brief Prepare intra-shard indices for shuffling.
        */
        void prepare_intra_shard_indices_() {
            // shuffle the examples inside the shards
            //if ( !intra_shard_indices_.empty() ) {
                // in case shards have different number of samples / sizes
            //    intra_shard_indices_.clear( );
            //}

            intra_shard_indices_.resize( shard_num_samples_ );

            init_identity_permutation( intra_shard_indices_.data(), (int)shard_num_samples_ );
            random_permutation( intra_shard_indices_.data(), (int)shard_num_samples_, &shuffle_rng_ );
        }

        /**
        * @brief Advance to the next shard of data.
        */
        void advance_() {
            if ( current_shard_idx_ == glob_result_.gl_pathc - 1 ) {
                // if we are at the last shard, we reset the loader and start a new epoch
                reset();
                return;
            }

            // advance the loader by loading the next data shard and resetting the position
            current_shard_idx_ = (current_shard_idx_ + 1) % glob_result_.gl_pathc;
            current_sample_idx_ = 0;
            load_shard_( (int)current_shard_idx_ );

            if ( should_shuffle_ ) {
                prepare_intra_shard_indices_();
            }
        }

        // variables related to distributed training
        // each process/worker has to access different parts of the data

        /**
        * @brief Rank of the current process.
        */
        int process_rank_;

        /**
        * @brief Total number of processes.
        */
        int num_processes_;

        // batch and token information...

        /**
        * @brief Batch size.
        */
        size_t batch_size_;

        /**
        * @brief Token size.
        */
        size_t token_size_;

        /**
        * @brief Total number of tokens.
        */
        size_t num_tokens_;

        /**
        * @brief Total number of samples in the current shard per process.
        */
        size_t shard_num_samples_;

        // shards and current position
        /**
        * @brief Stores the result of glob, for all shards we want to iterate.
        */
        Misc::glob_t glob_result_;

        /**
        * @brief The current shard we are reading from.
        */
        size_t current_shard_idx_;

        /**
        * @brief The current sample we are reading from.
        */
        size_t current_sample_idx_;

        // file handle
        /**
        * @brief File handle for tokens file.
        */
        std::ifstream tokens_file_;

        // data buffers
        /**
        * @brief Buffer to read data from file.
        */
        std::vector<uint16_t> buffer_;

        /**
        * @brief Input tokens into transformer.
        */
        Tensor<int> inputs_;

        /**
        * @brief Target tokens for the transformer.
        */
        Tensor<int> targets_;

        // random shuffle related variables
        /**
        * @brief Random number generator state for shuffling.
        */
        mt19937_state shuffle_rng_;

        /**
        * @brief Flag indicating whether to shuffle.
        */
        bool should_shuffle_;

        /**
        * @brief Indices of shards.
        */
        std::vector<int> shard_indices_;

        /**
        * @brief Indices within a shard.
        */
        std::vector<int> intra_shard_indices_;

        // sizes in bytes
        /**
        * @brief Total batch size in bytes across all processes.
        */
        size_t total_batch_size_bytes_;

        /**
        * @brief Inner-sample offset for this process in bytes.
        */
        size_t local_batch_offset_bytes_;

        /**
        * @brief Header size in bytes.
        */
        size_t header_bytes_;

        /**
        * @brief File size in bytes.
        */
        int64_t file_size_bytes_;
    };
} 