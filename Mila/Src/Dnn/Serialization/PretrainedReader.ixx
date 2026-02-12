/**
 * @file PretrainedReader.ixx
 * @brief Reader for Mila pretrained binary format.
 *
 * Provides direct access to pretrained model weights stored in Mila's
 * flat binary format. Used by fromPretrained() factory methods.
 */

module;
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <format>
#include <cstring>

export module Serialization.PretrainedReader;

import Serialization.Serializer;
import Serialization.OpenMode;
import Serialization.Tensor;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Compute.Device;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Metadata for a tensor blob in pretrained model format.
     */
    struct TensorBlobMetadata
    {
        std::string name;
        uint32_t dtype;
        std::vector<int64_t> shape;
        uint64_t offset;
        uint64_t nbytes;
    };

    /**
     * @brief Metadata for pretrained model.
     */
    export struct PretrainedMetadata
    {
        std::string architecture;
        std::string model_name;
        uint32_t vocab_size;
        uint32_t max_seq_length;
        uint32_t embedding_dim;
        uint32_t num_layers;
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t hidden_dim;
        bool use_bias;

        std::string activation;
        std::string norm_type;
        std::string attention_type;
        std::string positional_encoding;

        float rope_theta;
        float norm_epsilon;
    };

    enum class DType : uint32_t {
        Float32 = 0,
        Float16 = 1,
        BFloat16 = 2,
        Int32 = 3,
    };

    inline TensorDataType dtypeToTensorDataType( uint32_t dtype ) {
        switch ( static_cast<DType>(dtype) ) {
            case DType::Float32:  return TensorDataType::FP32;
            case DType::Float16:  return TensorDataType::FP16;
            case DType::BFloat16: return TensorDataType::BF16;
            case DType::Int32:    return TensorDataType::INT32;
            default:
                throw std::runtime_error( "Unknown dtype code: " + std::to_string( dtype ) );
        }
    }

    /**
     * @brief Reader for Mila pretrained binary format.
     *
     * File format:
     *  - Header: MILA magic (0x4D494C41), version, num_tensors
     *  - Metadata: JSON string with model configuration
     *  - Tensor index: for each tensor: name, dtype, shape, offset, nbytes
     *  - Tensor data: concatenated binary blobs
     *
     * Provides flat key-value access to tensors by name:
     *  - "lenc.wte.weight"
     *  - "tf_layer_0.ln_1.bias"
     *  - "ln_final.weight"
     *
     * Usage:
     * @code
     * PretrainedReader reader;
     * reader.open("gpt2_small.bin", OpenMode::Read);
     * 
     * auto metadata = reader.getPretrainedMetadata();
     * auto names = reader.getTensorNames();
     * 
     * for (const auto& name : names)
     * {
     *     auto tensor = reader.readTensor(name);
     *     network->loadTensorByFlatName(*tensor, name);
     * }
     * @endcode
     */
    export class PretrainedModelReader
    {
    public:

        /**
         * @brief Open a Mila model file for reading
         *
         * @param filepath Path to .bin model file
         * @throws std::runtime_error if file cannot be opened or format is invalid
         */
        explicit PretrainedModelReader( const std::filesystem::path& filepath )
            : filepath_( filepath )
        {
            file_.open( filepath, std::ios::binary );
            
            if ( !file_.is_open() )
            {
                throw std::runtime_error( "Cannot open pretrained model file: " + filepath.string() );
            }

            readHeader();
            readMetadata();
            readTensorIndex();
        }

        ~PretrainedModelReader()
        {
            close();
        }
        
        // ================================================================
        // Serializer interface
        // ================================================================

        bool close()
        {
            if (file_.is_open())
            {
                file_.close();
            }
            
            filename_.clear();
            tensor_index_.clear();
            
            return true;
        }

        bool isOpen() const noexcept
        {
            return file_.is_open();
        }

        const std::string& getFilename() const noexcept
        {
            return filename_;
        }

        // ================================================================
        // PretrainedReader-specific API
        // ================================================================

        /**
         * @brief Get pretrained model metadata.
         */
        const PretrainedMetadata& getPretrainedMetadata() const
        {
            return metadata_;
        }

        /**
         * @brief Get list of all tensor names in the model.
         */
        std::vector<std::string> getTensorNames() const
        {
            std::vector<std::string> names;
            names.reserve(tensor_index_.size());
            
            for (const auto& [name, _] : tensor_index_)
            {
                names.push_back(name);
            }
            
            return names;
        }

        /**
         * @brief Check if tensor exists.
         */
        bool hasTensor(const std::string& name) const
        {
            return tensor_index_.find(name) != tensor_index_.end();
        }

        /**
         * @brief Read raw tensor bytes by name.
         */
        TensorBlob readTensorBlob( const std::string& name )
        {
            const auto& blob_meta = getTensorBlobMetadata( name );

            std::vector<uint8_t> data( blob_meta.nbytes );

            file_.seekg( static_cast<std::streamoff>( blob_meta.offset) );
            file_.read(
                reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(blob_meta.nbytes) );

            if ( !file_.good() )
            {
                throw std::runtime_error( "Failed to read tensor: " + name );
            }

            // Convert to TensorMetadata
            TensorMetadata tensor_meta{
                .dtype = dtypeToTensorDataType( blob_meta.dtype ),
                .shape = blob_meta.shape,
                .total_bytes = blob_meta.nbytes
            };

            return TensorBlob{
                .metadata = tensor_meta,
                .data = std::move( data )
            };
        }

    private:

        static constexpr uint32_t MAGIC = 0x4D494C41;  // "MILA"
        static constexpr uint32_t VERSION = 1;

        std::filesystem::path filepath_;
        std::ifstream file_;
        std::string filename_;
        
        PretrainedMetadata metadata_;
        std::unordered_map<std::string, TensorBlobMetadata> tensor_index_;
        uint32_t num_tensors_{0};

        /**
        * @brief Get tensor metadata by name.
        */
        const TensorBlobMetadata& getTensorBlobMetadata( const std::string& name ) const
        {
            auto it = tensor_index_.find( name );
            if ( it == tensor_index_.end() )
            {
                throw std::runtime_error( "Tensor not found: " + name );
            }

            return it->second;
        }

        void readHeader()
        {
            uint32_t magic, version, num_tensors;

            file_.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            file_.read(reinterpret_cast<char*>(&version), sizeof(version));
            file_.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

            if (!file_.good())
            {
                throw std::runtime_error("Failed to read binary header");
            }

            if (magic != MAGIC)
            {
                throw std::runtime_error(
                    std::format("Invalid file format: wrong magic number (expected 0x{:X}, got 0x{:X})", 
                        MAGIC, magic));
            }

            if (version != VERSION)
            {
                throw std::runtime_error(
                    std::format("Unsupported file version: {}", version));
            }

            num_tensors_ = num_tensors;
        }

        void readMetadata()
        {
            uint32_t metadata_size;
            file_.read(reinterpret_cast<char*>(&metadata_size), sizeof(metadata_size));

            if (!file_.good() || metadata_size == 0)
            {
                throw std::runtime_error("Failed to read metadata size");
            }

            std::string json_str(metadata_size, '\0');
            file_.read(json_str.data(), static_cast<std::streamsize>(metadata_size));

            if (!file_.good())
            {
                throw std::runtime_error("Failed to read metadata JSON");
            }

            parseMetadataJSON(json_str);
        }

        void parseMetadataJSON(const std::string& json)
        {
            auto extract_string = [&](const std::string& key) -> std::string
            {
                auto pos = json.find("\"" + key + "\"");
                if (pos == std::string::npos) return "";

                auto value_start = json.find(":", pos) + 1;
                auto quote_start = json.find("\"", value_start);
                if (quote_start == std::string::npos) return "";

                auto quote_end = json.find("\"", quote_start + 1);
                if (quote_end == std::string::npos) return "";

                return json.substr(quote_start + 1, quote_end - quote_start - 1);
            };

            auto extract_int = [&](const std::string& key) -> uint32_t
            {
                auto pos = json.find("\"" + key + "\"");
                if (pos == std::string::npos) return 0;

                auto value_start = json.find(":", pos) + 1;
                auto comma_or_brace = json.find_first_of(",}", value_start);
                if (comma_or_brace == std::string::npos) return 0;

                std::string value_str = json.substr(value_start, comma_or_brace - value_start);
                value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
                value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);

                try
                {
                    return std::stoul(value_str);
                }
                catch (...)
                {
                    return 0;
                }
            };

            auto extract_bool = [&](const std::string& key) -> bool
            {
                auto pos = json.find("\"" + key + "\"");
                if (pos == std::string::npos) return false;

                auto true_pos = json.find("true", pos);
                auto false_pos = json.find("false", pos);

                return (true_pos != std::string::npos) && 
                       (false_pos == std::string::npos || true_pos < false_pos);
            };

            auto extract_float = [&](const std::string& key) -> float
            {
                auto pos = json.find("\"" + key + "\"");
                if (pos == std::string::npos) return 0.0f;

                auto value_start = json.find(":", pos) + 1;
                auto comma_or_brace = json.find_first_of(",}", value_start);
                if (comma_or_brace == std::string::npos) return 0.0f;

                std::string value_str = json.substr(value_start, comma_or_brace - value_start);
                value_str.erase(0, value_str.find_first_not_of(" \t\n\r"));
                value_str.erase(value_str.find_last_not_of(" \t\n\r") + 1);

                try
                {
                    return std::stof(value_str);
                }
                catch (...)
                {
                    return 0.0f;
                }
            };

            metadata_.architecture = extract_string("architecture");
            metadata_.model_name = extract_string("model_name");
            metadata_.vocab_size = extract_int("vocab_size");
            metadata_.max_seq_length = extract_int("max_seq_length");
            metadata_.embedding_dim = extract_int("embedding_dim");
            metadata_.num_layers = extract_int("num_layers");
            metadata_.num_heads = extract_int("num_heads");
            metadata_.num_kv_heads = extract_int("num_kv_heads");
            metadata_.hidden_dim = extract_int("hidden_dim");
            metadata_.use_bias = extract_bool("use_bias");
            metadata_.activation = extract_string("activation");
            metadata_.norm_type = extract_string("norm_type");
            metadata_.attention_type = extract_string("attention_type");
            metadata_.positional_encoding = extract_string("positional_encoding");
            metadata_.rope_theta = extract_float("rope_theta");
            metadata_.norm_epsilon = extract_float("norm_epsilon");
        }

        void readTensorIndex()
        {
            for (uint32_t i = 0; i < num_tensors_; ++i)
            {
                TensorBlobMetadata meta;

                uint32_t name_length;
                file_.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));

                if (!file_.good() || name_length == 0 || name_length > 1024)
                {
                    throw std::runtime_error(
                        std::format("Invalid tensor name length at index {}", i));
                }

                meta.name.resize(name_length);
                file_.read(meta.name.data(), static_cast<std::streamsize>(name_length));

                file_.read(reinterpret_cast<char*>(&meta.dtype), sizeof(meta.dtype));

                uint32_t ndim;
                file_.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

                if (!file_.good() || ndim > 8)
                {
                    throw std::runtime_error(
                        std::format("Invalid tensor dimensionality for '{}'", meta.name));
                }

                meta.shape.resize(ndim);

                for (uint32_t d = 0; d < ndim; ++d)
                {
                    uint32_t dim32;
                    file_.read(reinterpret_cast<char*>(&dim32), sizeof(dim32));
                    meta.shape[d] = static_cast<int64_t>(dim32);
                }

                file_.read(reinterpret_cast<char*>(&meta.offset), sizeof(meta.offset));
                file_.read(reinterpret_cast<char*>(&meta.nbytes), sizeof(meta.nbytes));

                if (!file_.good())
                {
                    throw std::runtime_error(
                        std::format("Failed to read tensor metadata for '{}'", meta.name));
                }

                tensor_index_[meta.name] = meta;
            }
        }
    };
}