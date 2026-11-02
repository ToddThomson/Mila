/**
 * @file InstalledModel.ixx
 * @brief A model in the local store, by name, with the facts an app needs before it loads anything.
 *
 * The one place a store name becomes a family, a weight format and a set of capabilities, so every app reads the
 * same answer. See Specifications/Deployment.md 8.1.
 */

export module Deployment.InstalledModel;

import std;
export import Dnn.ModelType;
export import Dnn.WeightQuantization;
export import Distribution.ModelCapabilities;
export import Distribution.StoredModel;
export import Data.BpeTokenizer;

import Dnn.Models.Architectures;
import Distribution.ModelRecord;
import Distribution.ModelStore;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Distribution;

    /**
     * @brief Whether this build runs a modality besides text. None yet: Gemma 4's image and audio paths are not built.
     */
    export inline constexpr bool isSupportedModality( std::string_view /*modality*/ ) noexcept
    {
        return false;
    }

    /**
     * @brief The attribution a licence requires wherever the model is presented, or empty.
     *
     * Llama 3.1 and 3.2 section 1.b.i require "Built with Llama" on a user interface among other places. Apache 2.0
     * requires the notice travel with the model, not that a user interface render one.
     */
    export inline std::string_view requiredAttribution( std::string_view license ) noexcept
    {
        return license.starts_with( "llama" ) ? std::string_view{ "Built with Llama" } : std::string_view{};
    }

    /**
     * @brief An installed model, looked up by its store name.
     *
     * Holds what the store recorded and answers in the library's types. The family and stored weight format here
     * come from the record, for an answer before the weights are opened; a load reads them from the weights header,
     * which is the authority.
     */
    export class InstalledModel
    {
    public:

        /// What the store answered for one model -- an install or a pull returns one -- in the library's types.
        explicit InstalledModel( StoredModel stored )
            : stored_( std::move( stored ) )
        {}

        /**
         * @brief Look a model up by name, as the store keys it (case-folded).
         *
         * @return Nothing when no model of that name is installed. A model whose files are missing is returned, and
         *         isComplete() says so, so the app can tell its user how to repair it.
         */
        static std::optional<InstalledModel> find( std::string_view name, const ModelStore& store = ModelStore{} )
        {
            const std::optional<ModelRecord> record = store.readRecord( std::string( name ) );

            if ( !record )
                return std::nullopt;

            return InstalledModel( store.describe( *record ) );
        }

        /// Every installed model, in the store's order.
        static std::vector<InstalledModel> list( const ModelStore& store = ModelStore{} )
        {
            std::vector<InstalledModel> models;

            for ( auto& stored : store.list() )
                models.push_back( InstalledModel( std::move( stored ) ) );

            return models;
        }

        /// The name as the store spells it, whatever case it was asked for in.
        [[nodiscard]] const std::string& name() const noexcept
        {
            return stored_.record.name;
        }

        /// The architecture the record declares, as its manifest spells it.
        [[nodiscard]] const std::string& architecture() const noexcept
        {
            return stored_.record.architecture;
        }

        /// The family the record declares; ModelType::Unknown for an architecture this build does not know.
        [[nodiscard]] ModelType family() const noexcept
        {
            return modelTypeFromString( stored_.record.architecture );
        }

        /// The weight format the record declares, or nothing for a scheme this build does not know.
        [[nodiscard]] std::optional<WeightQuantization> storedWeightQuantization() const noexcept
        {
            return parseWeightQuantization( stored_.record.weight_quantization );
        }

        /**
         * @brief What the model was trained to do, with only the modalities this build runs.
         *
         * A manifest written by a newer Mila may name a modality this one lacks; it is not reported.
         */
        [[nodiscard]] ModelCapabilities capabilities() const
        {
            ModelCapabilities supported = stored_.record.capabilities;

            std::erase_if( supported.modalities, []( const std::string& modality )
                {
                    return !isSupportedModality( modality );
                } );

            return supported;
        }

        /**
         * @brief The draft model to decode with, where the package carries one and this build drafts for its
         *        architecture.
         *
         * This is the MTP capability: present only when both hold, so the claim and the file cannot disagree.
         */
        [[nodiscard]] std::optional<std::filesystem::path> drafter() const
        {
            bool drafts = false;

            Architectures::visit( architecture(), [&]( auto named )
                {
                    drafts = decltype( named )::type::drafts;
                } );

            if ( stored_.drafter_path.empty() || !drafts )
                return std::nullopt;

            return stored_.drafter_path;
        }

        /// False when a file the record names is missing from the store, which makes the model unloadable.
        [[nodiscard]] bool isComplete() const noexcept
        {
            return stored_.complete;
        }

        [[nodiscard]] const std::filesystem::path& weightsPath() const noexcept
        {
            return stored_.weights_path;
        }

        [[nodiscard]] const std::filesystem::path& tokenizerPath() const noexcept
        {
            return stored_.tokenizer_path;
        }

        [[nodiscard]] const std::string& license() const noexcept
        {
            return stored_.record.license;
        }

        /// What the licence requires be shown wherever the model is presented, or empty.
        [[nodiscard]] std::string_view attribution() const noexcept
        {
            return requiredAttribution( stored_.record.license );
        }

        /// Everything the store recorded, for display: lineage, origin, files and their sizes.
        [[nodiscard]] const StoredModel& stored() const noexcept
        {
            return stored_;
        }

        /**
         * @brief The model's own tokenizer.
         *
         * @throws std::runtime_error when the architecture has no tokenizer in this build or the package carries none.
         */
        [[nodiscard]] std::shared_ptr<Data::BpeTokenizer> loadTokenizer() const
        {
            if ( stored_.tokenizer_path.empty() )
            {
                throw std::runtime_error( "'" + name() + "' carries no tokenizer" );
            }

            std::shared_ptr<Data::BpeTokenizer> tokenizer;

            Architectures::visit( architecture(), [&]( auto named )
                {
                    tokenizer = decltype( named )::type::loadTokenizer( stored_.tokenizer_path );
                } );

            if ( tokenizer )
            {
                return tokenizer;
            }

            throw std::runtime_error( "'" + name() + "' has architecture '" + architecture() +
                "', which has no tokenizer in this build of Mila" );
        }

    private:

        StoredModel stored_;
    };
}
