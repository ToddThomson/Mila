/**
 * @file Assistant.ResidentAgent.ixx
 * @brief The model the session has on the card, as an agent, or its session alone while none is.
 */

export module Assistant.ResidentAgent;

import std;
import Mila;

namespace Mila::Assistant
{
    using Mila::Deployment::DeploymentRequest;

    /**
     * @brief One agent at a time, and the session it holds, which outlives it (Direction.md 4.3).
     *
     * Opening another model, or the same one at another context, checks the name first and only then releases the one
     * resident, since the planner reads the card's free memory and must not find the old model still on it.
     */
    export class ResidentAgent
    {
    public:

        using Agent = Mila::AI::Agent;

        [[nodiscard]] bool resident() const noexcept
        {
            return agent_.has_value();
        }

        /// The agent on the card. Only while resident().
        [[nodiscard]] Agent& agent()
        {
            return *agent_;
        }

        [[nodiscard]] const Agent& agent() const
        {
            return *agent_;
        }

        /// The resident agent's session, or the one held while no model is.
        [[nodiscard]] Mila::AI::Session& session() noexcept
        {
            return agent_ ? agent_->session() : held_;
        }

        [[nodiscard]] const Mila::AI::Session& session() const noexcept
        {
            return agent_ ? agent_->session() : held_;
        }

        /// The resident model's name as the store spells it; empty while none is.
        [[nodiscard]] std::string name() const
        {
            return agent_ ? std::string( agent_->model().name() ) : std::string{};
        }

        /// The request the resident model was opened with. Only while resident().
        [[nodiscard]] const DeploymentRequest& request() const noexcept
        {
            return request_;
        }

        /// Free the card; the session, conversation and all, is held until the next model opens.
        void release()
        {
            if ( agent_ )
            {
                held_ = std::move( *agent_ ).release();
                agent_.reset();
            }
        }

        /**
         * @brief Open a model in place of the one resident, the session going with it.
         *
         * @throws Mila::AI::OpenRefusedError before anything is released: the resident model stays.
         * @throws std::exception for anything that fails after the release -- a model that does not fit the card, a
         *         file that will not read. Nothing is resident then, and the session is held.
         */
        void open( std::string_view name, const DeploymentRequest& request )
        {
            if ( auto refusal = Agent::check( name, request ) )
            {
                throw Mila::AI::OpenRefusedError( std::move( *refusal ) );
            }

            release();

            // The held session moves only once the model has opened, so a failure leaves it here.
            agent_.emplace( Agent::create( name, request, std::move( held_ ) ) );
            request_ = request;
        }

    private:

        std::optional<Agent> agent_;
        Mila::AI::Session held_;
        DeploymentRequest request_;
    };
}
