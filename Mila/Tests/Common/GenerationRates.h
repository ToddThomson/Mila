/**
 * @file GenerationRates.h
 * @brief Prefill and decode rates, timed by subtraction between two lengths so the load,
 * the language-model head and the single decode step cancel.
 *
 * Shared because a rate is only worth reading against another rate, and two models timed
 * by two methods are not comparable. Every arm -- packed codebook, FP4, any chassis --
 * goes through these two functions.
 */

#pragma once

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <stop_token>
#include <vector>

namespace Mila::Tests::Common
{
    /**
     * @brief Token ids that exist in every vocabulary in use, spread so no chassis sees a
     * degenerate repeated row. Values, not extents -- ids are out of dim_t's scope.
     *
     * `salt` shifts the whole stream so two prompts DIFFER AT TOKEN 0. That is not
     * cosmetic: a chassis with prompt-prefix reuse serves a prompt that extends a cached
     * one out of the KV cache and never prefills it, and then the subtraction below
     * measures nothing. Unsalted, Gemma 4 read -0.04 ms/token -- the long prompt "cost"
     * less than the short one because both were prefixes of the warm-up.
     */
    inline std::vector<std::int32_t> syntheticPrompt( std::int64_t length, int salt = 0 )
    {
        std::vector<std::int32_t> prompt;
        prompt.reserve( static_cast<std::size_t>( length ) );

        for ( std::int64_t i = 0; i < length; ++i )
            prompt.push_back( static_cast<std::int32_t>(
                1000 + ( i * 37 + salt * 613 ) % 4096 ) );

        return prompt;
    }

    /**
     * @brief Marginal seconds per prompt token, at max_new_tokens = 1.
     *
     * The difference between two prompt lengths removes everything that does not scale
     * with the prompt, which at one new token is the load, the head and the decode step.
     *
     * All three runs use different salts, so none is a prefix of another and no chassis
     * can serve one out of another's KV cache.
     */
    template<typename TModel>
    double prefillSecondsPerToken( TModel& model, std::int64_t long_length,
        std::int64_t short_length )
    {
        const std::vector<std::int32_t> warm_prompt = syntheticPrompt( long_length, 1 );
        const std::vector<std::int32_t> long_prompt = syntheticPrompt( long_length, 2 );
        const std::vector<std::int32_t> short_prompt = syntheticPrompt( short_length, 3 );

        auto once = [&]( const std::vector<std::int32_t>& prompt ) -> double
            {
                Mila::Dnn::GenerateParams params;
                params.max_new_tokens = 1;
                params.sampling.temperature = 0.0f;

                const auto start = std::chrono::steady_clock::now();
                model->generate( prompt, []( std::int32_t ) {}, params, std::stop_token{} );

                return std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start ).count();
            };

        // Warm: the first pass pays lazy cuBLASLt plan selection and first-touch paging.
        once( warm_prompt );

        const double long_seconds = once( long_prompt );
        const double short_seconds = once( short_prompt );

        return ( long_seconds - short_seconds )
            / static_cast<double>( long_length - short_length );
    }

    /**
     * @brief Marginal seconds per generated token, over one fixed prompt.
     *
     * The difference between two generation lengths removes the load and the prefill,
     * which is why this measures decode rather than a run that is mostly neither.
     */
    template<typename TModel>
    double decodeSecondsPerToken( TModel& model, const std::vector<std::int32_t>& prompt,
        int long_tokens, int short_tokens )
    {
        auto once = [&]( int tokens ) -> double
            {
                int produced = 0;

                Mila::Dnn::GenerateParams params;
                params.max_new_tokens = tokens;
                params.sampling.temperature = 0.0f;

                const auto start = std::chrono::steady_clock::now();
                const auto status = model->generate(
                    prompt, [&]( std::int32_t ) { ++produced; }, params, std::stop_token{} );
                const auto elapsed = std::chrono::steady_clock::now() - start;

                EXPECT_EQ( produced, tokens )
                    << "generation stopped early (" << to_string( status )
                    << "), which breaks the subtraction";

                return std::chrono::duration<double>( elapsed ).count();
            };

        once( short_tokens );

        const double short_seconds = once( short_tokens );
        const double long_seconds = once( long_tokens );

        return ( long_seconds - short_seconds )
            / static_cast<double>( long_tokens - short_tokens );
    }
}
