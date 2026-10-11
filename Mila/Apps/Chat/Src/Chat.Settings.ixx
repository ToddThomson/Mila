/**
 * @file Chat.Settings.ixx
 * @brief One settings document, merged key by key from ranked layers, remembering where each
 *        key came from.
 *
 * Implements the resolution order of Specifications/ChatConfiguration.md sections 3 and 4. The
 * origin record is not diagnostics: resolving a relative path and reporting a bad value both
 * have to know which layer wrote the key.
 */

module;
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

export module Chat.Settings;

import nlohmann.json;

namespace Mila::ChatApp
{
    /**
     * @brief Where a setting came from.
     *
     * Declaration order IS rank order -- later wins -- and the comparison operators are used to
     * ask whether a value came from above the compiled defaults. Inserting a layer in the middle
     * therefore changes what wins, which is the intent: the table in section 3 is the enum.
     */
    export enum class SettingsLayer
    {
        FamilyInvariants,      ///< 1: what the architecture can do. Compiled.
        ModelRecommendations,  ///< 2: what this checkpoint recommends. From its manifest.
        UserConfig,            ///< 3: how this person likes Chat.
        LocalConfig,           ///< 4: this checkout, or this image. The config that is FOUND.
        RememberedChoice,      ///< 5: the model last chosen from inside a session.
        CommandLine,           ///< 6: flags, and a file named with --settings.

        /// 7: set from inside the running session, by /context or /set. Outranks the command line
        /// because it is the most recent thing the user said, and is deliberately NOT persisted --
        /// a value that outranks a file the user authored must not also survive the process that
        /// heard it, or an experiment silently becomes a preference.
        SessionOverride,
    };

    export constexpr std::string_view layerName( SettingsLayer layer )
    {
        switch ( layer )
        {
            case SettingsLayer::FamilyInvariants:     return "the family defaults";
            case SettingsLayer::ModelRecommendations: return "the model record";
            case SettingsLayer::UserConfig:           return "the user config";
            case SettingsLayer::LocalConfig:          return "the local config";
            case SettingsLayer::RememberedChoice:     return "the remembered model choice";
            case SettingsLayer::CommandLine:          return "the command line";
            case SettingsLayer::SessionOverride:      return "this session";
        }

        return "the command line";
    }

    /**
     * @brief The layer that last wrote a key, and the file it wrote it from.
     *
     * @c file is empty for the layers that are not files. That distinction carries meaning: a
     * relative path set by a file resolves differently from one typed on the command line.
     */
    export struct SettingsOrigin
    {
        SettingsLayer layer{ SettingsLayer::FamilyInvariants };
        std::filesystem::path file;
    };

    /**
     * @brief One layer's contribution: an RFC 7386 merge patch.
     *
     * @c values must be a JSON object. A merge patch that is not one REPLACES what it is merged
     * into rather than amending it, so a file holding an array would not misconfigure one key --
     * it would erase every layer beneath it. Whoever reads a file checks that before it gets here.
     */
    export struct SettingsPatch
    {
        SettingsLayer layer{ SettingsLayer::FamilyInvariants };
        nlohmann::json values = nlohmann::json::object();
        std::filesystem::path file;
    };

    /**
     * @brief The merged settings, and the origin of every key in them.
     */
    export class MergedSettings
    {
    public:
        /**
         * @brief Merge one layer over what is already there.
         *
         * Origins are recorded for top-level keys, which is the whole of this document -- there
         * are no nested settings, and a nested one would need this to walk the patch.
         */
        void apply( const SettingsPatch& patch )
        {
            values_.merge_patch( patch.values );

            for ( const auto& entry : patch.values.items() )
            {
                // RFC 7386: a null DELETES the key rather than setting it to null, so the origin
                // goes with it. Leaving one behind would name a layer for a key nobody set.
                if ( entry.value().is_null() )
                {
                    origins_.erase( entry.key() );
                }
                else
                {
                    origins_[ entry.key() ] = SettingsOrigin{ patch.layer, patch.file };
                }
            }
        }

        /// Merge a run of layers in the order given, which must be rank order.
        void applyAll( const std::vector<SettingsPatch>& patches )
        {
            for ( const auto& patch : patches )
            {
                apply( patch );
            }
        }

        const nlohmann::json& values() const noexcept
        {
            return values_;
        }

        std::optional<SettingsOrigin> originOf( const std::string& key ) const
        {
            const auto found = origins_.find( key );

            if ( found == origins_.end() )
                return std::nullopt;

            return found->second;
        }

        /**
         * @brief Where a key came from, for a message a user can act on.
         *
         * A merged document has no single file to blame, so a complaint about a value that does
         * not name its source leaves the reader three files to search.
         */
        std::string describeOrigin( const std::string& key ) const
        {
            const auto origin = originOf( key );

            if ( !origin )
                return "unset";

            if ( !origin->file.empty() )
                return origin->file.string();

            return std::string( layerName( origin->layer ) );
        }

    private:
        nlohmann::json values_ = nlohmann::json::object();
        std::map<std::string, SettingsOrigin, std::less<>> origins_;
    };
}
