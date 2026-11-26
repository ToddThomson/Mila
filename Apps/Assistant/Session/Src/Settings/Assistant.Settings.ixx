/**
 * @file Assistant.Settings.ixx
 * @brief What the person chose in Mila Assistant's Settings, kept between launches (MilaAssistant.md 3.7).
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

        /// Whether Build & Code offers the experimental command tool (ContainedPrograms.md 4.11).
        bool run_commands{ false };

        /// The SHA-256 of the licence text the person accepted; empty until they have (AssistantOnboarding.md 4.1).
        std::string license_accepted;

        /// When they accepted it, ISO 8601 in UTC; empty until they have.
        std::string license_accepted_at;

        /**
         * @brief Where the onboarding stands (AssistantOnboarding.md 4.4): empty before it began, "started" from the
         *        onboarding model's pick until the interview ends, then "done".
         */
        std::string onboarding;

        /// The interview's chat, so a person who stops part way returns to it; empty when there is none.
        std::string interview_conversation;

        /// What the person asked to be called, from the interview; empty until they say.
        std::string person_name;

        /// What the person said they will mostly use Mila for, from the interview; empty until they say.
        std::string person_uses;

        /// What that work needs, as the model read the answer: conversation, files_and_tools or long_documents, each
        /// once, joined by commas (AssistantOnboarding.md 4.4, round 12); empty until it is read.
        std::string person_needs;

        /// The model the person declined to download when Mila asked, and when, so Mila can offer it again later from
        /// use (round 12); empty when nothing was declined.
        std::string declined_model;
        std::string declined_at;

        /// The model the person chose at the setup's end to move to, kept until it is installed and a new chat or
        /// launch runs on it (AssistantOnboarding.md 4.4, round 11); empty when there is none.
        std::string next_model;

        /// The person agreed to the onboarding model's download on *Getting Mila Ready* (AssistantOnboarding.md 4.3,
        /// round 9), so a download they stopped by quitting goes on at the next launch without asking again.
        bool download_agreed{ false };

        /// The version of Mila whose onboarding the person last went through; an update's own steps are due when it
        /// differs. Empty before the first.
        std::string last_version;

        Mila::Agents::SamplingOverride sampling;
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

    /**
     * @brief Whether the person went through the onboarding's panes in this version and the interview has not ended.
     *
     * Reaching the panes' end keeps the version, so a launch part way through the interview returns to its chat, not
     * to the panes (AssistantOnboarding.md 4.4).
     */
    export inline bool onboardingPanesGoneThrough( const Settings& settings, std::string_view version ) noexcept
    {
        return settings.onboarding == "started" && !settings.last_version.empty() && settings.last_version == version;
    }
}
