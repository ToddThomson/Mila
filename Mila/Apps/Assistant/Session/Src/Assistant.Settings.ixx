/**
 * @file Assistant.Settings.ixx
 * @brief What the person chose in the Mila app's Settings, kept between launches (MilaApp.md 3.7).
 */

export module Assistant.Settings;

import std;
import Mila;

namespace Mila::Assistant
{
    /**
     * @brief The session's settings as the person left them.
     *
     * Each one left empty is the model's own: its template's default for effort, its publisher's for sampling, and a
     * context the planner chooses for the card.
     */
    export struct Settings
    {
        /// The model last loaded, as the store spells it; empty when none has been.
        std::string model;

        /// A context fixed by the person; empty for the longest that fits.
        std::optional<Mila::Dnn::dim_t> context_length;

        /// Whether the model thinks, where its template has the switch.
        bool thinking{ true };

        /// One of the template's own effort values, where it has them; empty for its default.
        std::string effort;

        /// off, thoughts or all.
        std::string detail{ "off" };

        Mila::AI::SamplingOverride sampling;
    };

    /// The detail levels Settings offers, in order.
    export inline constexpr std::array<std::string_view, 3> kDetailLevels{ "off", "thoughts", "all" };

    /// The sampling settings Settings offers, by the names the protocol and the file use.
    export inline constexpr std::array<std::string_view, 4> kSamplingKeys{ "temperature", "top_k", "top_p", "presence_penalty" };

    /**
     * @brief Whether a value is one the sampler takes for this setting: the bounds the console's /set has.
     *
     * top_k must also be a whole number, which the caller checks, since a JSON number does not say.
     */
    export inline bool samplingValueAllowed( std::string_view key, double value ) noexcept
    {
        if ( key == "temperature" )
        {
            return value >= 0.0 && value <= 5.0;
        }

        if ( key == "top_k" )
        {
            return value >= 0.0 && value <= static_cast<double>( std::numeric_limits<int>::max() );
        }

        if ( key == "top_p" )
        {
            return value >= 0.0 && value <= 1.0;
        }

        if ( key == "presence_penalty" )
        {
            return value >= -2.0 && value <= 2.0;
        }

        return false;
    }
}
