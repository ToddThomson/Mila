module;
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

export module Dnn.ModelReader_Deprecated;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.Device;
import Compute.DeviceId;
import Compute.CpuMemoryResource;

// DEPRECATED: This is an older version of the ModelReader that reads the original Mila binary format.
// The PretrainedModelReader is the newer version

// REMOVE: after testing completed

//namespace Mila::Dnn
//{
//    /**
//     * @brief Metadata about a tensor in the model file
//     */
//    export struct TensorMetadata
//    {
//        std::string name;
//        uint32_t dtype;      // 0=float32, 1=float16, 2=bfloat16, 3=int32
//        std::vector<dim_t> shape;
//        uint64_t offset;     // Byte offset in file
//        uint64_t nbytes;     // Size in bytes
//    };
//
//    /**
//     * @brief Model metadata from file header
//     */
//    export struct ModelMetadata
//    {
//        std::string architecture;
//        std::string model_name;
//        uint32_t vocab_size;
//        uint32_t max_seq_length;
//        uint32_t embedding_dim;
//        uint32_t num_layers;
//        uint32_t num_heads;
//        uint32_t num_kv_heads;
//        uint32_t hidden_dim;
//        bool use_bias;
//        
//        std::string activation;
//        std::string norm_type;
//        std::string attention_type;
//        std::string positional_encoding;
//        
//        float rope_theta;
//        float norm_epsilon;
//
//        // Additional fields can be added as needed
//        std::unordered_map<std::string, std::string> extra_fields;
//    };
//
//    /**
//     * @brief Reader for Mila binary model format
//     */
//    export class ModelReader
//    {
//    public:
//        /**
//         * @brief Open a Mila model file for reading
//         *
//         * @param filepath Path to .bin model file
//         * @throws std::runtime_error if file cannot be opened or format is invalid
//         */
//        explicit ModelReader( const std::filesystem::path& filepath )
//            : filepath_( filepath )
//        {
//            file_.open( filepath, std::ios::binary );
//            if ( !file_.is_open() )
//            {
//                throw std::runtime_error( "Cannot open model file: " + filepath.string() );
//            }
//
//            readHeader();
//            readMetadata();
//            readTensorIndex();
//        }
//
//        ~ModelReader()
//        {
//            if ( file_.is_open() )
//            {
//                file_.close();
//            }
//        }
//
//        /**
//         * @brief Get model metadata
//         */
//        const ModelMetadata& getMetadata() const
//        {
//            return metadata_;
//        }
//
//        /**
//         * @brief Get list of all tensor names in the model
//         */
//        std::vector<std::string> getTensorNames() const
//        {
//            std::vector<std::string> names;
//            names.reserve( tensor_index_.size() );
//            for ( const auto& [name, meta] : tensor_index_ )
//            {
//                names.push_back( name );
//            }
//            return names;
//        }
//
//        /**
//         * @brief Check if a tensor exists in the model
//         */
//        bool hasTensor( const std::string& name ) const
//        {
//            return tensor_index_.find( name ) != tensor_index_.end();
//        }
//
//        /**
//         * @brief Get metadata for a specific tensor
//         */
//        const TensorMetadata& getTensorMetadata( const std::string& name ) const
//        {
//            auto it = tensor_index_.find( name );
//            if ( it == tensor_index_.end() )
//            {
//                throw std::runtime_error( "Tensor not found: " + name );
//            }
//            return it->second;
//        }
//
//        /**
//         * @brief Read a tensor's raw bytes
//         *
//         * @param name Tensor name
//         * @return Vector of bytes containing tensor data
//         */
//        std::vector<uint8_t> readTensorBytes( const std::string& name )
//        {
//            const auto& meta = getTensorMetadata( name );
//
//            std::vector<uint8_t> data( meta.nbytes );
//
//            file_.seekg( meta.offset );
//            file_.read( reinterpret_cast<char*>(data.data()), meta.nbytes );
//
//            if ( !file_.good() )
//            {
//                throw std::runtime_error( "Failed to read tensor: " + name );
//            }
//
//            return data;
//        }
//
//        /**
//         * @brief Read a tensor directly into a Tensor object
//         *
//         * @tparam TPrecision Target tensor precision
//         * @tparam MR Memory resource type
//         * @param name Tensor name
//         * @return Tensor containing the data (converted to TPrecision if needed)
//         */
//        std::unique_ptr<ITensor> readTensor( const std::string& name )
//        {
//            const auto& meta = getTensorMetadata( name );
//
//            auto bytes = readTensorBytes( name );
//
//            // Use host CPU memory resource for reader-created tensors.
//            Compute::DeviceId host_dev = Compute::Device::Cpu();
//
//            // Concrete host memory resource type
//            using HostMR = Compute::CpuMemoryResource;
//
//            switch ( meta.dtype )
//            {
//                case 0: // float32
//                {
//                    auto t = std::make_unique<Tensor<TensorDataType::FP32, HostMR>>( host_dev, meta.shape );
//                    convertAndCopyData<float>( bytes.data(), meta.dtype, t->data(), meta.nbytes );
//                    
//                    return t;
//                }
//                
//                // REVIEW: a host tensors cannot be float16 or bfloat16 directly
//                
//                //case 1: // float16 (stored as uint16_t)
//                //{
//                //    auto t = std::make_unique<Tensor<TensorDataType::FP16, HostMR>>( host_dev, meta.shape );
//                //    convertAndCopyData<uint16_t>( bytes.data(), meta.dtype, t->data(), meta.nbytes );
//                //    return t;
//                //}
//
//                //case 2: // bfloat16 (stored as uint16_t)
//                //{
//                //    auto t = std::make_unique<Tensor<TensorDataType::BF16, HostMR>>( host_dev, meta.shape );
//                //    convertAndCopyData<uint16_t>( bytes.data(), meta.dtype, t->data(), meta.nbytes );
//                //    return t;
//                //}
//
//                case 3: // int32
//                {
//                    auto t = std::make_unique<Tensor<TensorDataType::INT32, HostMR>>( host_dev, meta.shape );
//                    convertAndCopyData<int32_t>( bytes.data(), meta.dtype, t->data(), meta.nbytes );
//                    return t;
//                }
//
//                default:
//                    throw std::runtime_error( "Unsupported dtype in model file: " + std::to_string( meta.dtype ) );
//            }
//        }
//
//    private:
//        static constexpr uint32_t MAGIC = 0x4D494C41;  // "MILA"
//        static constexpr uint32_t VERSION = 1;
//
//        std::filesystem::path filepath_;
//        std::ifstream file_;
//        ModelMetadata metadata_;
//        std::unordered_map<std::string, TensorMetadata> tensor_index_;
//        uint32_t num_tensors_ = 0;
//
//        /**
//         * @brief Read and validate file header
//         */
//        void readHeader()
//        {
//            uint32_t magic, version, num_tensors;
//
//            file_.read( reinterpret_cast<char*>(&magic), sizeof( magic ) );
//            file_.read( reinterpret_cast<char*>(&version), sizeof( version ) );
//            file_.read( reinterpret_cast<char*>(&num_tensors), sizeof( num_tensors ) );
//
//            if ( magic != MAGIC )
//            {
//                throw std::runtime_error( "Invalid file format: wrong magic number" );
//            }
//
//            if ( version != VERSION )
//            {
//                throw std::runtime_error( "Unsupported file version: " + std::to_string( version ) );
//            }
//
//            num_tensors_ = num_tensors;
//        }
//
//        /**
//         * @brief Read and parse JSON metadata
//         */
//        void readMetadata()
//        {
//            uint32_t metadata_size;
//            file_.read( reinterpret_cast<char*>(&metadata_size), sizeof( metadata_size ) );
//
//            std::string json_str( metadata_size, '\0' );
//            file_.read( json_str.data(), metadata_size );
//
//            // Parse JSON (you could use nlohmann/json or a simple parser)
//            // For now, this is a placeholder - implement based on your JSON library choice
//            parseMetadataJSON( json_str );
//        }
//
//        /**
//         * @brief Parse metadata JSON string
//         *
//         * NOTE: This is a simplified version. You should use a proper JSON library
//         * like nlohmann/json or implement a simple key-value parser.
//         */
//        void parseMetadataJSON( const std::string& json )
//        {
//            // TODO: Implement proper JSON parsing
//            // For now, extract key fields with simple string parsing
//            // In production, use nlohmann/json or similar
//
//            // Example simplified parsing (replace with real JSON parser):
//            auto extract_string = [&]( const std::string& key ) -> std::string {
//                auto pos = json.find( "\"" + key + "\"" );
//                if ( pos == std::string::npos ) return "";
//
//                auto value_start = json.find( ":", pos ) + 1;
//                auto quote_start = json.find( "\"", value_start );
//                auto quote_end = json.find( "\"", quote_start + 1 );
//
//                return json.substr( quote_start + 1, quote_end - quote_start - 1 );
//                };
//
//            auto extract_int = [&]( const std::string& key ) -> uint32_t {
//                auto pos = json.find( "\"" + key + "\"" );
//                if ( pos == std::string::npos ) return 0;
//
//                auto value_start = json.find( ":", pos ) + 1;
//                auto comma_or_brace = json.find_first_of( ",}", value_start );
//
//                std::string value_str = json.substr( value_start, comma_or_brace - value_start );
//                // Trim whitespace
//                value_str.erase( 0, value_str.find_first_not_of( " \t\n\r" ) );
//                value_str.erase( value_str.find_last_not_of( " \t\n\r" ) + 1 );
//
//                return std::stoul( value_str );
//                };
//
//            auto extract_bool = [&]( const std::string& key ) -> bool {
//                auto pos = json.find( "\"" + key + "\"" );
//                if ( pos == std::string::npos ) return false;
//                return json.find( "true", pos ) < json.find( "false", pos );
//                };
//
//            auto extract_float = [&]( const std::string& key ) -> float {
//                auto pos = json.find( "\"" + key + "\"" );
//                if ( pos == std::string::npos ) return 0.0f;
//
//                auto value_start = json.find( ":", pos ) + 1;
//                auto comma_or_brace = json.find_first_of( ",}", value_start );
//
//                std::string value_str = json.substr( value_start, comma_or_brace - value_start );
//                value_str.erase( 0, value_str.find_first_not_of( " \t\n\r" ) );
//                value_str.erase( value_str.find_last_not_of( " \t\n\r" ) + 1 );
//
//                return std::stof( value_str );
//                };
//
//            // Extract metadata fields
//            metadata_.architecture = extract_string( "architecture" );
//            metadata_.model_name = extract_string( "model_name" );
//            metadata_.vocab_size = extract_int( "vocab_size" );
//            metadata_.max_seq_length = extract_int( "max_seq_length" );
//            metadata_.embedding_dim = extract_int( "embedding_dim" );
//            metadata_.num_layers = extract_int( "num_layers" );
//            metadata_.num_heads = extract_int( "num_heads" );
//            metadata_.num_kv_heads = extract_int( "num_kv_heads" );
//            metadata_.hidden_dim = extract_int( "hidden_dim" );
//            metadata_.use_bias = extract_bool( "use_bias" );
//            metadata_.activation = extract_string( "activation" );
//            metadata_.norm_type = extract_string( "norm_type" );
//            metadata_.attention_type = extract_string( "attention_type" );
//            metadata_.positional_encoding = extract_string( "positional_encoding" );
//            metadata_.rope_theta = extract_float( "rope_theta" );
//            metadata_.norm_epsilon = extract_float( "norm_epsilon" );
//        }
//
//        /**
//         * @brief Read tensor index (names, shapes, offsets)
//         */
//        void readTensorIndex()
//        {
//            for ( uint32_t i = 0; i < num_tensors_; ++i )
//            {
//                TensorMetadata meta;
//
//                // Read name
//                uint32_t name_length;
//                file_.read( reinterpret_cast<char*>( &name_length ), sizeof( name_length ) );
//                meta.name.resize( name_length );
//                file_.read( meta.name.data(), name_length );
//
//                // Read dtype
//                file_.read( reinterpret_cast<char*>(&meta.dtype), sizeof( meta.dtype ) );
//
//                // Read shape (on-disk dims are stored as uint32_t; convert to dim_t)
//                uint32_t ndim;
//                file_.read( reinterpret_cast<char*>(&ndim), sizeof( ndim ) );
//
//                meta.shape.resize( ndim );
//
//                for ( uint32_t d = 0; d < ndim; ++d )
//                {
//                    uint32_t dim32;
//                    file_.read( reinterpret_cast<char*>( &dim32 ), sizeof( dim32 ) );
//                    meta.shape[ d ] = static_cast<dim_t>( dim32 );
//                }
//
//                // Read offset and size
//                file_.read( reinterpret_cast<char*>(&meta.offset), sizeof( meta.offset ) );
//                file_.read( reinterpret_cast<char*>(&meta.nbytes), sizeof( meta.nbytes ) );
//
//                tensor_index_[ meta.name ] = meta;
//            }
//        }
//
//        /**
//         * @brief Convert tensor data from file dtype to target dtype
//         */
//        template<typename TPrecision>
//        void convertAndCopyData( const void* src, uint32_t src_dtype,
//            TPrecision* dst, size_t nbytes )
//        {
//            size_t num_elements = nbytes / getDtypeSize( src_dtype );
//
//            switch ( src_dtype )
//            {
//                case 0:  // float32
//                    convertData( static_cast<const float*>(src), dst, num_elements );
//                    break;
//                case 1:  // float16
//                    convertData( static_cast<const uint16_t*>(src), dst, num_elements );
//                    break;
//                case 2:  // bfloat16
//                    convertDataBF16( static_cast<const uint16_t*>(src), dst, num_elements );
//                    break;
//                case 3:  // int32
//                    convertData( static_cast<const int32_t*>(src), dst, num_elements );
//                    break;
//                default:
//                    throw std::runtime_error( "Unsupported dtype: " + std::to_string( src_dtype ) );
//            }
//        }
//
//        /**
//         * @brief Get size in bytes for a dtype code
//         */
//        static size_t getDtypeSize( uint32_t dtype )
//        {
//            switch ( dtype )
//            {
//                case 0: return sizeof( float );      // float32
//                case 1: return sizeof( uint16_t );   // float16
//                case 2: return sizeof( uint16_t );   // bfloat16
//                case 3: return sizeof( int32_t );    // int32
//                default: throw std::runtime_error( "Unknown dtype" );
//            }
//        }
//
//        /**
//         * @brief Convert between numeric types
//         */
//        template<typename SrcT, typename DstT>
//        void convertData( const SrcT* src, DstT* dst, size_t count )
//        {
//            for ( size_t i = 0; i < count; ++i )
//            {
//                dst[ i ] = static_cast<DstT>( src[ i ] );
//            }
//        }
//
//        /**
//         * @brief Convert bfloat16 (stored as uint16) to target type
//         */
//        template<typename DstT>
//        void convertDataBF16( const uint16_t* src, DstT* dst, size_t count )
//        {
//            for ( size_t i = 0; i < count; ++i )
//            {
//                // BF16 to FP32: shift left 16 bits
//                uint32_t fp32_bits = static_cast<uint32_t>( src[ i ] ) << 16;
//                float fp32_value;
//                std::memcpy( &fp32_value, &fp32_bits, sizeof( float ) );
//                
//                dst[ i ] = static_cast<DstT>( fp32_value );
//            }
//        }
//    };
//}