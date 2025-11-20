module;
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <format>
#include <sstream>

export module InferenceEngine;

import Dnn.Tensor;

namespace Mila::Dnn::Engine
{
    /**
     * @brief Lightweight inference engine for exported models
     */
    export class InferenceEngine
    {
    public:
        /**
         * @brief Load an exported model from file
         */
        explicit InferenceEngine( const std::filesystem::path& model_path )
        {
            load_model( model_path );
        }

        /**
         * @brief Prevent copying (models can be large)
         */
        InferenceEngine( const InferenceEngine& ) = delete;
        InferenceEngine& operator=( const InferenceEngine& ) = delete;

        /**
         * @brief Allow moving
         */
        InferenceEngine( InferenceEngine&& ) noexcept = default;
        InferenceEngine& operator=( InferenceEngine&& ) noexcept = default;

        /**
         * @brief Execute inference on input data
         * @param input Input tensor
         * @return Output tensor
         */
        Tensor predict( const Tensor& input )
        {
            if (!m_network)
            {
                throw std::runtime_error( "InferenceEngine: Model not loaded" );
            }

            // Validate input shape if specs are available
            if (!m_metadata.inputs.empty())
            {
                validate_input( input, 0 );
            }

            // Execute forward pass (no gradient computation)
            Tensor output = m_network->forward( input );

            return output;
        }

        /**
         * @brief Execute inference with multiple inputs
         * @param inputs Vector of input tensors
         * @return Vector of output tensors
         */
        std::vector<Tensor> predict( const std::vector<Tensor>& inputs )
        {
            if (!m_network)
            {
                throw std::runtime_error( "InferenceEngine: Model not loaded" );
            }

            // Validate number of inputs
            if (!m_metadata.inputs.empty() && inputs.size() != m_metadata.inputs.size())
            {
                throw std::runtime_error(
                    std::format( "InferenceEngine: Expected {} inputs, got {}",
                        m_metadata.inputs.size(), inputs.size() )
                );
            }

            // For now, assume single input/output model
            // Multi-input models would need different network API
            if (inputs.size() != 1)
            {
                throw std::runtime_error(
                    "InferenceEngine: Multi-input models not yet supported"
                );
            }

            return { predict( inputs[0] ) };
        }

        /**
         * @brief Batch inference - process multiple samples
         * @param batch Vector of input tensors (one per sample)
         * @return Vector of output tensors (one per sample)
         */
        std::vector<Tensor> predict_batch( const std::vector<Tensor>& batch )
        {
            std::vector<Tensor> outputs;
            outputs.reserve( batch.size() );

            for (const auto& input : batch)
            {
                outputs.push_back( predict( input ) );
            }

            return outputs;
        }

        /**
         * @brief Get model metadata
         */
        const ModelMetadata& metadata() const
        {
            return m_metadata;
        }

        /**
         * @brief Get input specifications
         */
        const std::vector<ModelMetadata::TensorSpec>& input_specs() const
        {
            return m_metadata.inputs;
        }

        /**
         * @brief Get output specifications
         */
        const std::vector<ModelMetadata::TensorSpec>& output_specs() const
        {
            return m_metadata.outputs;
        }

        /**
         * @brief Get the underlying network (for advanced use)
         */
        const Network& network() const
        {
            if (!m_network)
            {
                throw std::runtime_error( "InferenceEngine: Model not loaded" );
            }
            return *m_network;
        }

        /**
         * @brief Get model information as string
         */
        std::string info() const
        {
            std::ostringstream oss;
            oss << "Model: " << m_metadata.name << "\n";
            oss << "Version: " << m_metadata.version << "\n";
            oss << "Framework: " << m_metadata.framework_version << "\n";
            oss << "Training Epochs: " << m_metadata.training_epochs << "\n";
            oss << "Final Loss: " << m_metadata.final_loss << "\n";

            if (!m_metadata.inputs.empty())
            {
                oss << "\nInputs:\n";
                for (size_t i = 0; i < m_metadata.inputs.size(); ++i)
                {
                    const auto& spec = m_metadata.inputs[i];
                    oss << "  [" << i << "] " << spec.name << " (";
                    oss << spec.dtype << ", shape: [";
                    for (size_t j = 0; j < spec.shape.size(); ++j)
                    {
                        oss << spec.shape[j];
                        if (j < spec.shape.size() - 1) oss << ", ";
                    }
                    oss << "])\n";
                }
            }

            if (!m_metadata.outputs.empty())
            {
                oss << "\nOutputs:\n";
                for (size_t i = 0; i < m_metadata.outputs.size(); ++i)
                {
                    const auto& spec = m_metadata.outputs[i];
                    oss << "  [" << i << "] " << spec.name << " (";
                    oss << spec.dtype << ", shape: [";
                    for (size_t j = 0; j < spec.shape.size(); ++j)
                    {
                        oss << spec.shape[j];
                        if (j < spec.shape.size() - 1) oss << ", ";
                    }
                    oss << "])\n";
                }
            }

            return oss.str();
        }

        /**
         * @brief Set the device for inference (CPU, CUDA, etc.)
         */
        void set_device( const std::string& device )
        {
            if (!m_network)
            {
                throw std::runtime_error( "InferenceEngine: Model not loaded" );
            }

            // Move network to specified device
            // This would need to be implemented in your Network class
            // m_network->to_device(device);

            m_device = device;
        }

        /**
         * @brief Get current device
         */
        const std::string& device() const
        {
            return m_device;
        }

    private:
        ModelMetadata m_metadata;
        std::unique_ptr<Network> m_network;
        std::string m_device = "CPU";
        std::filesystem::path m_model_path;

        /**
         * @brief Load model from file
         */
        void load_model( const std::filesystem::path& model_path )
        {
            m_model_path = model_path;

            if (!std::filesystem::exists( model_path ))
            {
                throw std::runtime_error(
                    std::format( "InferenceEngine: Model file not found: {}",
                        model_path.string() )
                );
            }

            ZipSerializer serializer;
            if (!serializer.openForRead( model_path.string() ))
            {
                throw std::runtime_error(
                    std::format( "InferenceEngine: Failed to open model: {}",
                        model_path.string() )
                );
            }

            try
            {
                // Validate format
                auto format = serializer.getMetadata( "format" );
                if (format != "mila-inference-v1")
                {
                    throw std::runtime_error(
                        std::format( "InferenceEngine: Unsupported model format: {}", format )
                    );
                }

                // Load metadata from JSON
                if (!serializer.hasFile( "metadata.json" ))
                {
                    throw std::runtime_error(
                        "InferenceEngine: Model missing metadata.json"
                    );
                }

                auto metadata_size = serializer.getFileSize( "metadata.json" );
                std::string metadata_json( metadata_size, '\0' );
                serializer.extractData( "metadata.json", metadata_json.data(), metadata_size );

                json meta_j = json::parse( metadata_json );
                m_metadata = ModelMetadata::from_json( meta_j );

                // Load network architecture
                if (!serializer.hasFile( "architecture.json" ))
                {
                    throw std::runtime_error(
                        "InferenceEngine: Model missing architecture.json"
                    );
                }

                auto arch_size = serializer.getFileSize( "architecture.json" );
                std::string arch_json( arch_size, '\0' );
                serializer.extractData( "architecture.json", arch_json.data(), arch_size );

                // Reconstruct network from architecture
                m_network = Network::from_json( arch_json );

                if (!m_network)
                {
                    throw std::runtime_error(
                        "InferenceEngine: Failed to reconstruct network from architecture"
                    );
                }

                // Load weights
                m_network->load( serializer );

                // Set network to evaluation mode (no dropout, batch norm in eval mode, etc.)
                m_network->eval();

                serializer.close();

                std::println( "? Model loaded: {} (trained for {} epochs, final loss: {:.6f})",
                    m_metadata.name,
                    m_metadata.training_epochs,
                    m_metadata.final_loss );

            }
            catch (const json::exception& e)
            {
                serializer.close();
                throw std::runtime_error(
                    std::format( "InferenceEngine: JSON parse error: {}", e.what() )
                );
            }
            catch (const std::exception& e)
            {
                serializer.close();
                throw std::runtime_error(
                    std::format( "InferenceEngine: Failed to load model: {}", e.what() )
                );
            }
        }

        /**
         * @brief Validate input tensor against specification
         */
        void validate_input( const Tensor& input, size_t input_idx ) const
        {
            if (input_idx >= m_metadata.inputs.size())
            {
                return; // No spec available
            }

            const auto& spec = m_metadata.inputs[input_idx];

            // Check shape (allowing batch dimension flexibility)
            if (input.shape().size() != spec.shape.size())
            {
                // Allow for batch dimension
                if (input.shape().size() != spec.shape.size() + 1)
                {
                    throw std::runtime_error(
                        std::format( "InferenceEngine: Input shape mismatch. "
                            "Expected {} dimensions, got {}",
                            spec.shape.size(), input.shape().size() )
                    );
                }
            }

            // Validate non-batch dimensions
            size_t offset = (input.shape().size() == spec.shape.size() + 1) ? 1 : 0;
            for (size_t i = 0; i < spec.shape.size(); ++i)
            {
                if (spec.shape[i] != 0 && input.shape()[i + offset] != spec.shape[i])
                {
                    throw std::runtime_error(
                        std::format( "InferenceEngine: Input shape mismatch at dimension {}. "
                            "Expected {}, got {}",
                            i, spec.shape[i], input.shape()[i + offset] )
                    );
                }
            }

            // Check dtype if available
            if (!spec.dtype.empty() && input.dtype() != spec.dtype)
            {
                std::println( stderr,
                    "Warning: Input dtype mismatch. Expected {}, got {}",
                    spec.dtype, input.dtype() );
            }
        }
    };
}