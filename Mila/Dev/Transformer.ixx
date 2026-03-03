//module;
//#include <stdexcept>
//#include <algorithm>
//#include <vector>
//#include <cmath>
//#include <random>
//
//export module Dnn.Transformer;
//
//import Dnn.GenerateParams;
//import Dnn.Tensor;
//import Dnn.TensorDataType;
//import Dnn.Network;
//import Compute.Device;
//import Compute.DeviceType;
//import Compute.DeviceTypeTraits;
//import Compute.CpuMemoryResource;
//import Data.Tokenizer;
//
//namespace Mila::Dnn
//{
//    using TokenId = Data::TokenId;
//    using namespace Mila::Dnn::Compute;
//
//    export template<DeviceType TDeviceType, TensorDataType TPrecision>
//        class Transformer : public Network<TDeviceType, TPrecision>
//    {
//    public:
//
//        using DeviceMR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
//
//        virtual Tensor<TensorDataType::INT32, CpuMemoryResource> generate(
//            const Tensor<TensorDataType::INT32, CpuMemoryResource>& prompt_tokens,
//            const GenerateParams& params ) final
//        {
//            if ( this->isTraining() ) {
//                throw std::runtime_error( "generate() requires inference mode" );
//            }
//
//            int64_t prompt_len = prompt_tokens.shape()[ 0 ];
//            int64_t max_total_len = prompt_len + params.max_new_tokens;
//
//            // Output buffer
//            Tensor<TensorDataType::INT32, CpuMemoryResource> output(
//                Device::Cpu(), { max_total_len } );
//            std::copy_n( prompt_tokens.data(), prompt_len, output.data() );
//
//            // Initialize KV cache (architecture-specific)
//            if ( !kv_cache_initialized_ || current_cache_len_ < max_total_len ) {
//                initializeKVCache( max_total_len );  // Virtual call
//                kv_cache_initialized_ = true;
//            }
//
//            // Transfer prompt to device
//            Tensor<TensorDataType::INT32, DeviceMR> prompt_device(
//                this->device_id_, prompt_tokens.shape() );
//            copy( prompt_tokens, prompt_device );
//
//            // PREFILL: Process entire prompt
//            auto& prefill_logits = forwardPrefill( prompt_device, prompt_len );  // Virtual
//            current_cache_len_ = prompt_len;
//
//            // DECODE: Generate tokens one by one
//            Tensor<TensorDataType::INT32, DeviceMR> single_token_device(
//                this->device_id_, { 1 } );
//
//            for ( int64_t i = 0; i < params.max_new_tokens; ++i ) {
//                // Get logits for next token prediction
//                auto& logits = (i == 0) ? prefill_logits :
//                    forwardDecode( single_token_device, current_cache_len_ );  // Virtual
//
//                // Sample next token (shared logic)
//                TokenId next_token = sampleToken( logits, params, current_cache_len_ - 1 );
//                output.data()[ prompt_len + i ] = next_token;
//
//                // Early stopping
//                // FIXME:
//                /*if ( params.eos_token_id && next_token == *params.eos_token_id ) {
//                    return output.slice( { 0 }, { prompt_len + i + 1 } );
//                }*/
//
//                // Prepare for next iteration
//                single_token_device.data()[ 0 ] = next_token;
//                current_cache_len_++;
//            }
//
//            return output;
//        }
//
//        virtual void resetKVCache();
//
//
//        virtual void setMaxCacheLength( int64_t max_len );
//
//    protected:
//
//        // Architecture-specific hooks (pure virtual)
//        virtual Tensor<TPrecision, DeviceMR>& forwardPrefill(
//            const Tensor<TensorDataType::INT32, DeviceMR>& tokens,
//            int64_t seq_len ) = 0;
//
//        virtual Tensor<TPrecision, DeviceMR>& forwardDecode(
//            const Tensor<TensorDataType::INT32, DeviceMR>& token,
//            int64_t position ) = 0;
//
//        // Subclasses implement their own cache init
//        virtual void initializeKVCache( int64_t max_len ) = 0;
//
//        virtual void clearKVCache() = 0;
//
//        // Shared state
//        bool kv_cache_initialized_{ false };
//        int64_t current_cache_len_{ 0 };
//    };
//}