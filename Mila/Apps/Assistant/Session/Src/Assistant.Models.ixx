/**
 * @file Assistant.Models.ixx
 * @brief The models view's work in the store and at the publisher, and its sentences when a model will not open.
 */

export module Assistant.Models;

import std;
import nlohmann.json;
import Mila;

namespace Mila::Assistant
{
    using Mila::Dnn::WeightQuantization;
    using Mila::Deployment::DeploymentRefusal;
    using Mila::Deployment::InstalledModel;

    /// A size as the models view shows it.
    export inline std::string formatBytes( std::uint64_t bytes )
    {
        constexpr double kGigabyte = 1024.0 * 1024.0 * 1024.0;
        constexpr double kMegabyte = 1024.0 * 1024.0;

        if ( bytes >= static_cast<std::uint64_t>( kGigabyte ) )
        {
            return std::format( "{:.2f} GB", static_cast<double>( bytes ) / kGigabyte );
        }

        return std::format( "{:.1f} MB", static_cast<double>( bytes ) / kMegabyte );
    }

    /// A weight format by the word the models view uses for it.
    export constexpr std::string_view quantizationWord( WeightQuantization quantization ) noexcept
    {
        switch ( quantization )
        {
            case WeightQuantization::FP8: return "fp8";
            case WeightQuantization::FP4: return "fp4";
            case WeightQuantization::Q4_0: return "q4_0";
            case WeightQuantization::Plan: return "cb2-3";
            case WeightQuantization::Fp4Codebook3: return "fp4-cb3";
            case WeightQuantization::None: return "none";
        }

        return "none";
    }

    /// A quantization a load may ask for (MilaApp.md 4.2), or nothing for any other word.
    export constexpr std::optional<WeightQuantization> parseLoadQuantization( std::string_view word ) noexcept
    {
        if ( word == "none" ) return WeightQuantization::None;
        if ( word == "fp8" ) return WeightQuantization::FP8;
        if ( word == "fp4" ) return WeightQuantization::FP4;
        if ( word == "q4_0" ) return WeightQuantization::Q4_0;

        return std::nullopt;
    }

    /// Every installed model, as `models.list` carries it.
    export inline nlohmann::json installedModels()
    {
        nlohmann::json installed = nlohmann::json::array();

        for ( const InstalledModel& model : InstalledModel::list() )
        {
            const std::optional<WeightQuantization> stored = model.storedWeightQuantization();

            std::string family = Mila::Dnn::toString( model.family() );
            std::ranges::transform( family, family.begin(), []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

            installed.push_back( {
                { "name", model.name() },
                { "size_bytes", model.stored().bytes_on_disk },
                { "quantization", stored ? std::string( quantizationWord( *stored ) ) : model.stored().record.weight_quantization },
                { "family", family } } );
        }

        return installed;
    }

    /**
     * @brief Every model the publisher offers, as `models.list` carries it: one small request for the list and one
     *        for each model's manifest, which holds its size. A model whose manifest cannot be read is listed by name.
     *
     * @throws std::exception when the publisher cannot be listed at all.
     */
    export inline nlohmann::json publishedModels()
    {
        nlohmann::json published = nlohmann::json::array();

        if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
        {
            return published;
        }

        const std::string owner( Mila::Distribution::kDefaultHubOwner );
        const auto hub = Mila::Distribution::makeDefaultModelHub();
        const auto installed = Mila::Distribution::ModelStore{}.list();

        for ( const auto& model : hub->listModels( owner ) )
        {
            std::uint64_t bytes = 0;
            std::string quantization;

            if ( model.hasManifest() )
            {
                try
                {
                    const auto manifest = Mila::Distribution::parseModelManifest(
                        hub->fetchManifest( Mila::Distribution::ModelCoordinate{ owner, model.repository } ), model.repository );

                    for ( const auto& file : manifest.files )
                    {
                        bytes += file.bytes;
                    }

                    const auto parsed = Mila::Dnn::parseWeightQuantization( manifest.weight_quantization );
                    quantization = parsed ? std::string( quantizationWord( *parsed ) ) : manifest.weight_quantization;
                }
                catch ( const std::exception& )
                {
                    // One model's manifest unread costs that row its size, not the whole list.
                }
            }

            // One repository is one model at one format, so its name is the store's.
            const bool is_installed = std::ranges::any_of( installed,
                [&model]( const auto& stored ) { return stored.record.name == model.repository; } );

            published.push_back( {
                { "name", model.repository },
                { "size_bytes", bytes },
                { "quantization", quantization },
                { "installed", is_installed } } );
        }

        return published;
    }

    /**
     * @brief Install a published model by name, reporting each file's bytes as they arrive.
     *
     * @return What to tell the person: what was installed, and its licence when it has one.
     * @throws std::exception when the install fails; the store keeps nothing of it.
     */
    export inline std::vector<std::string> installModel( const std::string& name,
        const std::function<void( std::uint64_t received, std::uint64_t total )>& progress )
    {
        if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
        {
            throw std::runtime_error( "this build of Mila cannot download" );
        }

        Mila::Distribution::ModelStore store;

        const auto hub = Mila::Distribution::makeDefaultModelHub( [&progress]( std::uint64_t received, std::uint64_t total )
            {
                if ( total > 0 )
                {
                    progress( received, total );
                }

                return true;
            } );

        Mila::Distribution::ModelResolver resolver( store, *hub );
        const auto pulled = resolver.pull( name, std::string( Mila::Distribution::kDefaultHubOwner ) );

        std::vector<std::string> lines;
        lines.push_back( std::format( "Installed {} ({}).", pulled.record.name, formatBytes( pulled.bytes_on_disk ) ) );

        if ( !pulled.record.license.empty() )
        {
            lines.push_back( std::format( "License: {}. The terms are published with the model.", pulled.record.license ) );
        }

        return lines;
    }

    /**
     * @brief Remove an installed model, deleting only the files no other installed model shares.
     *
     * @return What to tell the person, the first line saying whether it was removed.
     */
    export inline std::vector<std::string> removeModel( const std::string& requested_name )
    {
        Mila::Distribution::ModelStore store;

        const std::optional<InstalledModel> installed = InstalledModel::find( requested_name, store );
        const std::string name = installed ? installed->name() : requested_name;
        const auto report = store.remove( name );

        std::vector<std::string> lines;

        if ( report.records_removed == 0 && report.retained.empty() )
        {
            lines.push_back( std::format( "{} is not installed.", name ) );

            return lines;
        }

        if ( report.records_removed == 0 )
        {
            lines.push_back( std::format( "{} was not removed: some of its files are in use. Close whatever has it loaded, "
                "then remove it again.", name ) );
        } else {
            lines.push_back( std::format( "Removed {}, {} freed.", name, formatBytes( report.bytes_reclaimed ) ) );
        }

        if ( report.records_removed == 1 && report.blobs_removed == 0 && report.unreadable_records.empty() )
        {
            lines.push_back( "Its files are shared with another installed model, so none were deleted." );
        }

        for ( const auto& retained : report.retained )
        {
            lines.push_back( std::format( "Could not delete {}, which may be in use.", retained ) );
        }

        return lines;
    }

    /// Why a model will not open, said to the person: the library's reason, and what to do where there is something.
    export inline std::string refusalText( const Mila::AI::OpenRefusal& refusal )
    {
        std::string text = refusal.detail;

        if ( !text.empty() )
        {
            text.front() = static_cast<char>( std::toupper( static_cast<unsigned char>( text.front() ) ) );
        }

        if ( !text.ends_with( '.' ) )
        {
            text += '.';
        }

        if ( refusal.reason == Mila::AI::OpenRefusal::Reason::FilesMissing
            || refusal.reason == Mila::AI::OpenRefusal::Reason::ChatTemplateNotRecorded )
        {
            text += " Remove it and install it again.";
        }

        return text;
    }

    /**
     * @brief Why a model does not fit the card, said to the person.
     *
     * @param quantizable Whether its weights are full precision, so quantized as they load they would need less.
     */
    export inline std::string refusalText( const DeploymentRefusal& refusal, bool quantizable )
    {
        const std::string free = formatBytes( refusal.reading().free_bytes );

        switch ( refusal.reason() )
        {
            case DeploymentRefusal::Reason::WeightsExceedDevice:
                return std::format( "Its weights need {} and this GPU has {} free.{}",
                    formatBytes( refusal.footprint().device_parameter_bytes ), free,
                    quantizable ? std::string( " Quantized to fp4 as it loads, it needs less." ) : std::string{} );

            case DeploymentRefusal::Reason::FixedContextDoesNotFit:
                return std::format( "A context of {} needs {} and this GPU has {} free. An automatic context picks the "
                    "longest that fits.", refusal.contextLength(), formatBytes( refusal.footprint().totalDeviceBytes() ), free );

            case DeploymentRefusal::Reason::NothingAboveTheFloorFits:
                return std::format( "Even a context of {} needs {} and this GPU has {} free.",
                    refusal.contextLength(), formatBytes( refusal.footprint().totalDeviceBytes() ), free );

            case DeploymentRefusal::Reason::DeviceDoesNotReportMemory:
                return "This GPU does not report its memory, so no context can be measured for it. Choose one yourself.";
        }

        return refusal.toString();
    }
}
