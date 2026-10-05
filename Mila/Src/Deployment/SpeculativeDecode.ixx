/**
 * @file SpeculativeDecode.ixx
 * @brief A deployment's draft model: the weights that propose tokens and how many it proposes a round.
 *
 * See Specifications/Gemma4Mtp.md section 4.1 and Deployment.md section 2.1.
 */

module;
#include <filesystem>
#include <format>
#include <stdexcept>

export module Deployment.SpeculativeDecode;

import Dnn.TensorTypes;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;

    /**
     * @brief Decode with a draft model: each round it proposes `draft_tokens` tokens, the model checks them in one
     *        pass and keeps the ones it agrees with.
     *
     * The draft model is built beside the model and priced with it; a deployment that does not select one builds
     * nothing for it.
     */
    export struct SpeculativeDecode
    {
        /// The most tokens one round may propose: the model checks them and the token before in one pass of at most 8.
        static constexpr dim_t kMaximumDraftTokens = 7;

        /// The draft model's weights file.
        std::filesystem::path draft_model;

        /// Tokens the draft model proposes a round, 1 to kMaximumDraftTokens.
        dim_t draft_tokens{ 0 };

        /// @throws std::invalid_argument when the draft length is outside 1 to kMaximumDraftTokens.
        void validate() const
        {
            if ( draft_tokens < 1 || draft_tokens > kMaximumDraftTokens )
            {
                throw std::invalid_argument( std::format(
                    "SpeculativeDecode: draft_tokens {} must be 1 to {}", draft_tokens, kMaximumDraftTokens ) );
            }
        }

        bool operator==( const SpeculativeDecode& ) const = default;
    };
}
