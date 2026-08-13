/**
 * @file main.cpp
 * @brief Mila C++ quick start: one prompt in, generated tokens streamed out.
 *
 * The smallest complete program that runs a local LLM with Mila -- load a model from the
 * store, encode a prompt, and stream the reply as it is produced. Single-shot by design:
 * there is no conversation history and no REPL, because neither teaches anything about
 * Mila. The chat harness (Mila/Adaptors/Chat) is where multi-turn, channel routing and
 * tool calls live.
 *
 * The Python counterpart at ../Python/generate.py does the same thing, with the same
 * model and the same prompt, so the two can be read side by side.
 */

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <vector>

// Both of these are WORKAROUNDS for a module-consumption defect, not requirements of the
// API -- and both must stay BEFORE `import Mila;` (importing first fails outright with a
// fatal MSVC modules error). Instantiating a Mila model instantiates its vtable, and
// Component::toString() is virtual, so GemmaModel::toString()'s body is compiled here;
// it uses std::ostringstream, whose definition does not reach this translation unit from
// the module. Without <sstream>: "'oss' uses undefined class std::basic_ostringstream".
// Delete these once the library stops requiring them.
#include <sstream>

import Mila;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Data;

namespace
{
    /// The published flagship. Install it with the chat harness (/install) or from Python;
    /// loading never downloads, so an uninstalled name is an error rather than a surprise
    /// multi-gigabyte transfer.
    constexpr const char* kModelName = "gemma-4-12b-it-fp4";

    /// Well under the model's ceiling. Context length drives KV-cache VRAM, and a first run
    /// should fit comfortably rather than probe the limit.
    constexpr dim_t kContextLength = 4096;

    /// Bounds the loop so the sample always terminates, even on a model that will not stop.
    constexpr int kMaxNewTokens = 512;

    /// Gemma collapses to a single system instruction. A first run answers noticeably
    /// better with one than without.
    constexpr const char* kSystemPrompt = "You are a helpful assistant.";

    /**
     * @brief Wrap one user message in the Gemma 4 instruct template.
     *
     * Gemma 4 frames each turn as <|turn>{role}\n{content}<turn|>\n, opens with <bos>, and
     * primes generation with a bare <|turn>model\n. The tokenizer registers those control
     * tokens, so they are written as literal text and encode as one atomic token each.
     * (NOT the Gemma 3 style <start_of_turn>/<end_of_turn> -- this vocabulary has neither.)
     *
     * Thinking is OFF, and that takes two things, not one. Omitting the <|think|> trigger
     * from a system turn deactivates it -- but the 12B then emits "ghost" thought sections
     * anyway, so an EMPTY <|channel>thought<channel|> is primed onto the prompt to suppress
     * them. That prime is load-bearing: without it the model narrates at you. It applies to
     * the 12B/26B/31B sizes; with thinking ON you must not prime it, because that pre-empts
     * the model's own reasoning. Chat::formatGemmaPrompt is the full version of this.
     */
    std::string buildGemmaPrompt( const std::string& user_message )
    {
        std::string prompt( Gemma::kBos );

        prompt += std::string( Gemma::kTurnOpen ) + "system\n" + kSystemPrompt
            + std::string( Gemma::kTurnClose ) + "\n";
        prompt += std::string( Gemma::kTurnOpen ) + "user\n" + user_message
            + std::string( Gemma::kTurnClose ) + "\n";
        prompt += std::string( Gemma::kTurnOpen ) + "model\n";
        prompt += std::string( Gemma::kChannelOpen ) + "thought\n"
            + std::string( Gemma::kChannelClose );

        return prompt;
    }

    /// Everything after the program name, joined -- so `mila_quickstart why is the sky blue`
    /// works without quoting. Empty when nothing was passed.
    std::string promptFromArguments( int argc, char** argv )
    {
        std::string joined;

        for ( int i = 1; i < argc; ++i )
        {
            if ( !joined.empty() )
                joined += ' ';

            joined += argv[i];
        }

        return joined;
    }
}

int main( int argc, char** argv )
{
    try
    {
        // Silent by default (NullSink); pass a ConsoleSink to opt in to log output.
        Mila::initialize();

        // Printed before anything touches the GPU or the store, so a failure below is
        // separable from a failure to build and link Mila at all.
        std::cout << "Mila " << Mila::getAPIVersion().toString() << "\n";

        std::string user_message = promptFromArguments( argc, argv );

        if ( user_message.empty() )
        {
            std::cout << "Prompt: " << std::flush;

            // std::fgets, not std::getline or std::cin.getline. WORKAROUND: any C++ stream
            // INPUT in a translation unit that imports Mila fails to compile -- both forms
            // instantiate basic_istream machinery here and hit "'_Ok' uses undefined class
            // basic_istream::sentry". Verified by compiling the identical getline call with
            // and without `import Mila;`: without it, clean. Output (std::cout) is fine.
            // <cstdio> instantiates nothing, so it sidesteps the defect entirely.
            char line[ 4096 ]{};

            if ( std::fgets( line, sizeof( line ), stdin ) == nullptr )
            {
                std::cerr << "No prompt given.\n";
                Mila::shutdown();

                return EXIT_FAILURE;
            }

            user_message = line;

            // fgets keeps the newline; trim it and any trailing whitespace.
            while ( !user_message.empty()
                && ( user_message.back() == '\n' || user_message.back() == '\r'
                    || user_message.back() == ' ' ) )
            {
                user_message.pop_back();
            }

            if ( user_message.empty() )
            {
                std::cerr << "No prompt given.\n";
                Mila::shutdown();

                return EXIT_FAILURE;
            }
        }

        // The store is the only source: nothing here consults a hub or accepts a path.
        Mila::Distribution::ModelStore store;
        const auto installed = store.locate( kModelName );

        if ( !installed.has_value() || !installed->complete )
        {
            std::cerr << "'" << kModelName << "' is not installed.\n"
                      << "Install it from Python:\n"
                      << "  import mila\n"
                      << "  mila.initialize(\"warning\")\n"
                      << "  mila.ModelStore().pull(\"" << kModelName
                      << "\", mila.default_hub_owner())\n";
            Mila::shutdown();

            return EXIT_FAILURE;
        }

        std::cout << "Loading " << kModelName << " ...\n" << std::flush;

        auto tokenizer = BpeTokenizer::loadGemma( installed->tokenizer_path );

        GemmaModelConfig model_config( kContextLength );
        model_config.withFP4Quantization();

        auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
            installed->weights_path, model_config );

        // TokenId is int32_t, which is what generate() takes, so this needs no conversion.
        const std::vector<TokenId> prompt_tokens =
            tokenizer->encode( buildGemmaPrompt( user_message ) );

        GenerateParams params;
        params.max_new_tokens = kMaxNewTokens;

        std::cout << "\n";

        // The model owns the decode loop and pushes each token here on this thread, so
        // decoding one id at a time is what makes the reply appear as it is produced.
        // Flushing every token is the whole point -- buffered output would arrive at once.
        //
        // Decoding per token is safe here because decode() yields BYTES: a token can carry
        // part of a multi-byte code point, and the bytes simply concatenate into a correct
        // UTF-8 stream. The Python sample has to buffer ids until they decode cleanly only
        // because Python decodes to str, which cannot hold a partial code point.
        const auto status = model->generate(
            prompt_tokens,
            [&]( int32_t token_id ) {
                std::cout << tokenizer->decode( std::span( &token_id, 1 ) ) << std::flush;
            },
            params );

        // Why it stopped is the one outcome a caller cannot reconstruct from the tokens.
        std::cout << "\n\n[" << to_string( status ) << "]\n";

        Mila::shutdown();

        return EXIT_SUCCESS;
    }
    catch ( const std::exception& error )
    {
        std::cerr << "\nError: " << error.what() << "\n";

        return EXIT_FAILURE;
    }
}
