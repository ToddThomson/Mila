/**
 * @file Assistant.SetupSteps.ixx
 * @brief What the setup has to accomplish, which the app keeps and the interview's tools mark (AssistantOnboarding.md
 *        4.4, rounds 11 and 13).
 */

export module Assistant.SetupSteps;

import std;
import nlohmann.json;

namespace Mila::Assistant
{
    /**
     * @brief The setup's objectives, each pending, done or skipped: what to call the person and what they will mostly
     *        use Mila for.
     *
     * Kept with the setup's chat under its own key, never as the chat's plan: the setup is not work the person
     * approved, and nothing of it is drawn in the chat. The workspace reads what is still outstanding to put an
     * unfinished setup under *Needs you*. A skip is an answer, so the setup ends once each is done or skipped.
     */
    export class SetupSteps
    {
    public:

        enum class Step
        {
            Name,
            Uses,
        };

        /// Every objective pending.
        static SetupSteps fresh()
        {
            return SetupSteps{};
        }

        /// The objectives as toJson() wrote them; anything else is a fresh start.
        static SetupSteps fromJson( const nlohmann::json& stored )
        {
            SetupSteps steps;

            if ( !stored.is_array() || stored.size() != kTitles.size() )
            {
                return steps;
            }

            for ( std::size_t index = 0; index < kTitles.size(); ++index )
            {
                const nlohmann::json& kept = stored[ index ];

                if ( !kept.is_object() || kept.value( "objective", std::string{} ) != kTitles[ index ] )
                {
                    return SetupSteps{};
                }

                const std::string status = kept.value( "status", std::string{} );

                if ( status != "pending" && status != "done" && status != "skipped" )
                {
                    return SetupSteps{};
                }

                steps.statuses_[ index ] = status;
                steps.said_[ index ] = kept.value( "said", std::string{} );
            }

            return steps;
        }

        /**
         * @brief The objectives still outstanding in what a setup chat stores, as the workspace names them; null for a
         *        chat that is not the setup's.
         */
        static nlohmann::json outstandingIn( const nlohmann::json& stored )
        {
            // Objectives an earlier build stored under other names are no setup this one can finish, so they are not
            // flagged as one.
            if ( !stored.is_array() || stored.size() != kTitles.size() || fromJson( stored ).toJson() != stored )
            {
                return nullptr;
            }

            nlohmann::json outstanding = nlohmann::json::array();

            for ( const std::string& title : fromJson( stored ).outstanding() )
            {
                outstanding.push_back( title );
            }

            return outstanding;
        }

        /// The objective done, with what the person said.
        void done( Step step, std::string_view said )
        {
            const std::size_t index = indexOf( step );

            statuses_[ index ] = "done";
            said_[ index ] = std::string( said );
        }

        /// The objective skipped; a skip is an answer, so it is no longer outstanding.
        void skip( Step step )
        {
            const std::size_t index = indexOf( step );

            statuses_[ index ] = "skipped";
            said_[ index ].clear();
        }

        [[nodiscard]] bool pending( Step step ) const
        {
            return statuses_[ indexOf( step ) ] == "pending";
        }

        /// Whether the setup is over: every objective done or skipped.
        [[nodiscard]] bool finished() const
        {
            return !pending( Step::Name ) && !pending( Step::Uses );
        }

        /// The objectives still pending, by their names.
        [[nodiscard]] std::vector<std::string> outstanding() const
        {
            std::vector<std::string> titles;

            for ( std::size_t index = 0; index < kTitles.size(); ++index )
            {
                if ( statuses_[ index ] == "pending" )
                {
                    titles.emplace_back( kTitles[ index ] );
                }
            }

            return titles;
        }

        /// As the setup's chat stores them: [{objective, status, said}].
        [[nodiscard]] nlohmann::json toJson() const
        {
            nlohmann::json stored = nlohmann::json::array();

            for ( std::size_t index = 0; index < kTitles.size(); ++index )
            {
                stored.push_back( { { "objective", kTitles[ index ] }, { "status", statuses_[ index ] },
                    { "said", said_[ index ] } } );
            }

            return stored;
        }

    private:

        static constexpr std::array<std::string_view, 2> kTitles{ "What to call you", "About you, and what you would like help with" };

        SetupSteps()
        {
            statuses_.fill( "pending" );
        }

        static constexpr std::size_t indexOf( Step step ) noexcept
        {
            return static_cast<std::size_t>( step );
        }

        std::array<std::string, 2> statuses_;
        std::array<std::string, 2> said_;
    };
}
