/**
 * @file Agent.Cuda.cpp
 * @brief One program runs Gemma, Llama and Qwen by changing only the name, streaming and calling a tool; every call
 *        is asked, declined, refused or stopped as ToolCalling.md 3.3 says, a response keeps to its limits (4), and a
 *        session outlives its model (Direction.md 4.3).
 *
 * Reads installed models from the model store, read only, and skips a family with none installed.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <source_location>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using namespace Mila::Distribution;
    using namespace Mila::AI;

    namespace
    {
        constexpr dim_t kContextLength = 8192;

        constexpr const char* kWeatherParameters =
            R"({"type":"object","properties":{"city":{"type":"string","description":"The city, for example Paris"}},)"
            R"("required":["city"]})";

        constexpr const char* kWeatherResult = "Sunny, 17 degrees Celsius.";

        /// A tool whose calls are counted and whose arguments are kept.
        Tool weatherTool( std::vector<std::string>& arguments_seen )
        {
            return Tool{
                "get_weather",
                "Get the current weather in a city.",
                kWeatherParameters,
                [&arguments_seen]( std::string_view arguments, const ToolGrant&, std::stop_token )
                {
                    arguments_seen.emplace_back( arguments );

                    return std::string( kWeatherResult );
                } };
        }

        DeploymentRequest requestOn( int device_index )
        {
            DeploymentRequest request;
            request.withDevice( DeviceId{ DeviceType::Cuda, device_index } );
            request.withContextLength( kContextLength );

            return request;
        }

        /// The first of these names the store has, or nothing.
        std::optional<std::string> firstInstalled( std::initializer_list<const char*> names )
        {
            for ( const char* name : names )
            {
                if ( InstalledModel::find( name ).has_value() )
                {
                    return std::string( name );
                }
            }

            return std::nullopt;
        }

        /// The first CUDA device that holds the model at the test's context, or nothing.
        std::optional<DeploymentRequest> requestThatFits( const std::string& name )
        {
            for ( int device = 0; device < getDeviceCount( DeviceType::Cuda ); ++device )
            {
                const DeploymentRequest request = requestOn( device );

                if ( Agent::Handle::planFor( name, request ).has_value() )
                {
                    return request;
                }
            }

            return std::nullopt;
        }

        /// Why a model cannot be run here, or nothing when it can.
        std::optional<std::string> whyNotRunnable( const std::optional<std::string>& name )
        {
            if ( !name )
            {
                return std::format( "none of this family's models is installed in {}", ModelStore().root().string() );
            }

            const InstalledModel installed = *InstalledModel::find( *name );

            if ( installed.architecture() == "llama" && installed.stored().record.architecture_version.empty() )
            {
                return std::format( "'{}' was installed before architecture_version existed; reinstall it", *name );
            }

            if ( !requestThatFits( *name ) )
            {
                return std::format( "no CUDA device holds '{}' at context {}", *name, kContextLength );
            }

            return std::nullopt;
        }

        /**
         * @brief The program the gate is about: a model named, a tool registered, a question streamed.
         *
         * Everything the families do differently stays behind the name.
         */
        template<typename TModel>
        void askForTheWeather( const std::string& name )
        {
            std::vector<std::string> arguments_seen;
            std::vector<std::string> asked;

            auto agent = Agent::create( name, *requestThatFits( name ), Session{}
                .withTool( weatherTool( arguments_seen ) )
                .onAsk( [&asked]( const ToolQuestion& question )
                    {
                        asked.emplace_back( question.name );

                        return ToolAnswer::Once;
                    } ) );

            agent.model().model().seedSampler( 7 );

            std::string streamed_text;
            std::string streamed_reasoning;
            std::vector<AgentEvent::Kind> kinds;
            std::vector<std::string> results;

            const Response response = agent.respond( "What is the weather in Paris right now?",
                [&]( const AgentEvent& event )
                {
                    kinds.push_back( event.kind );

                    if ( event.kind == AgentEvent::Kind::Answer )
                    {
                        streamed_text += event.text;
                    }
                    else if ( event.kind == AgentEvent::Kind::Reasoning )
                    {
                        streamed_reasoning += event.text;
                    }
                    else if ( event.kind == AgentEvent::Kind::ToolCalled )
                    {
                        EXPECT_EQ( event.call->name, "get_weather" );
                        EXPECT_FALSE( event.is_error ) << event.text;
                        results.emplace_back( event.text );
                    }
                } );

            // The agent's part, whatever the model: the call is made, asked about, run, and its result reaches the model.
            ASSERT_FALSE( arguments_seen.empty() ) << name << " answered without calling the tool: " << response.text;
            EXPECT_NE( arguments_seen.front().find( "Paris" ), std::string::npos ) << arguments_seen.front();
            EXPECT_EQ( asked.size(), arguments_seen.size() ) << name << ": a call ran without being asked about";
            EXPECT_EQ( results.size(), arguments_seen.size() );

            EXPECT_EQ( response.status, GenerateStatus::Success ) << to_string( response.status );
            EXPECT_GE( response.rounds, 2 );
            EXPECT_EQ( streamed_text, response.text );
            EXPECT_EQ( streamed_reasoning, response.reasoning );

            ASSERT_FALSE( kinds.empty() );
            EXPECT_EQ( kinds.front(), AgentEvent::Kind::RoundBegan );

            const auto conversation = agent.session().conversation();

            ASSERT_GE( conversation.size(), 3u );
            EXPECT_EQ( conversation.front().role, Conversation::Role::User );
            EXPECT_FALSE( conversation[ 1 ].tool_calls.empty() );
            EXPECT_EQ( conversation[ 2 ].role, Conversation::Role::Tool );
            EXPECT_EQ( conversation[ 2 ].tool_name, "get_weather" );
            EXPECT_EQ( conversation[ 2 ].content, kWeatherResult );

            // Descent: the objects that ran.
            EXPECT_NE( agent.model().template as<TModel>(), nullptr ) << name;
            EXPECT_EQ( agent.plan().contextLength(), kContextLength );

            const bool agentic = agent.model().capabilities().agentic;

            ::testing::Test::RecordProperty( "model", name );
            ::testing::Test::RecordProperty( "agentic", agentic ? "true" : "false" );
            ::testing::Test::RecordProperty( "rounds", response.rounds );

            // The model's part, asked only of a model Mila measured as an agent: it answers from the result.
            if ( agentic )
            {
                EXPECT_EQ( kinds.back(), AgentEvent::Kind::RoundEnded );
                EXPECT_EQ( conversation.back().role, Conversation::Role::Assistant );
                EXPECT_TRUE( conversation.back().tool_calls.empty() );
                EXPECT_NE( response.text.find( "17" ), std::string::npos ) << name << ": " << response.text;
            }
        }

        /**
         * @brief Every prompt prefix a model reports it reused, matched to the round that reused it.
         *
         * Read from the model's log line, the one report of it there is; every record is passed on to the logger
         * this one stands in for while it lives.
         */
        class ReuseLog final : public Mila::Logging::Logger
        {
        public:

            struct Prefill
            {
                dim_t prompt{ 0 };
                dim_t skipped{ 0 };
            };

            ReuseLog()
                : previous_( Logger::defaultLogger() )
            {
                Logger::setDefaultLogger( this );
            }

            ~ReuseLog() override
            {
                Logger::setDefaultLogger( &previous_ );
            }

            /// Called as each round begins.
            void beginRound()
            {
                rounds_.emplace_back();
            }

            /// Each round's reuse, or nothing for a round that prefilled its whole prompt.
            const std::vector<std::optional<Prefill>>& rounds() const noexcept
            {
                return rounds_;
            }

            void setLevel( Mila::Logging::LogLevel level ) override { previous_.setLevel( level ); }
            Mila::Logging::LogLevel getLevel() const override { return previous_.getLevel(); }
            bool isEnabled( Mila::Logging::LogLevel level ) const override { return previous_.isEnabled( level ); }

            void log_trace( std::string_view message, const std::source_location& location ) override
            {
                previous_.log_trace( message, location );
            }

            void log_debug( std::string_view message, const std::source_location& location ) override
            {
                previous_.log_debug( message, location );
            }

            void log_info( std::string_view message, const std::source_location& location ) override
            {
                record( message );
                previous_.log_info( message, location );
            }

            void log_warning( std::string_view message, const std::source_location& location ) override
            {
                previous_.log_warning( message, location );
            }

            void log_error( std::string_view message, const std::source_location& location ) override
            {
                previous_.log_error( message, location );
            }

            void log_critical( std::string_view message, const std::source_location& location ) override
            {
                previous_.log_critical( message, location );
            }

            void log( std::string_view message, Mila::Logging::LogLevel level, const std::source_location& location ) override
            {
                if ( level == Mila::Logging::LogLevel::Info )
                {
                    record( message );
                }

                previous_.log( message, level, location );
            }

        private:

            void record( std::string_view message )
            {
                static const std::regex reuse( R"(KV prefix reuse -- skipped (\d+) of (\d+) prompt tokens)" );

                std::match_results<std::string_view::const_iterator> match;

                if ( !rounds_.empty() && std::regex_search( message.begin(), message.end(), match, reuse ) )
                {
                    rounds_.back() = Prefill{ std::stoll( match[ 2 ].str() ), std::stoll( match[ 1 ].str() ) };
                }
            }

            Logger& previous_;
            std::vector<std::optional<Prefill>> rounds_;
        };

        /**
         * @brief The gate on what a conversation prefills: within a response a round prefills only the results it
         *        adds, framed as the template frames them; the next message never prefills from the start.
         *
         * Two messages, each asking for a call. Rounds are matched to the calls they answered by walking the
         * conversation, which holds each round's reply and then its results.
         */
        void expectEachRoundPrefillsOnlyWhatItAdds( const std::string& name )
        {
            std::vector<std::string> arguments_seen;

            auto agent = Agent::create( name, *requestThatFits( name ), Session{}
                .withTool( weatherTool( arguments_seen ) )
                .onAsk( []( const ToolQuestion& ) { return ToolAnswer::Once; } ) );

            agent.model().model().seedSampler( 7 );

            ReuseLog reuse;

            const std::vector<std::string> messages{
                "What is the weather in Paris right now?", "And what is the weather in Rome right now?" };

            for ( std::size_t message = 0; message < messages.size(); ++message )
            {
                const std::size_t first_round = reuse.rounds().size();
                const std::size_t first_turn = agent.session().conversation().size();

                // Each round's prompt is the last one's, the tokens it generated and its results: none dropped.
                dim_t expected_prompt = static_cast<dim_t>(
                    agent.model().tokenizer().encode( agent.prompt( messages[ message ] ).text ).size() );
                std::vector<dim_t> generated;

                const Response response = agent.respond( messages[ message ], [&reuse, &generated]( const AgentEvent& event )
                    {
                        if ( event.kind == AgentEvent::Kind::RoundBegan )
                        {
                            reuse.beginRound();
                            generated.push_back( 0 );
                        }
                        else if ( event.kind == AgentEvent::Kind::Token )
                        {
                            ++generated.back();
                        }
                    } );

                if ( message == 0 )
                {
                    ASSERT_GE( response.rounds, 2 ) << name << " answered without calling the tool: " << response.text;
                }
                else
                {
                    // The first message's rounds are in the model; this one resumes from them.
                    const auto& resumed = reuse.rounds()[ first_round ];

                    ASSERT_TRUE( resumed.has_value() ) << name << ": the next message prefilled the conversation from its start";
                    std::printf( "%s\n", std::format( "{}: next message prefilled {} of {} prompt tokens",
                        name, resumed->prompt - resumed->skipped, resumed->prompt ).c_str() );
                }

                ASSERT_EQ( reuse.rounds().size(), first_round + static_cast<std::size_t>( response.rounds ) );

                const auto conversation = agent.session().conversation();
                std::size_t turn = first_turn + 1;

                for ( int round = 1; round < response.rounds; ++round )
                {
                    // The reply that made the calls, then their results.
                    ASSERT_LT( turn, conversation.size() );
                    ASSERT_FALSE( conversation[ turn ].tool_calls.empty() );

                    const std::size_t first_result = ++turn;

                    while ( turn < conversation.size() && conversation[ turn ].role == Conversation::Role::Tool )
                    {
                        ++turn;
                    }

                    const Conversation::Prompt after = agent.model().protocol()->continueWithToolResults(
                        conversation.subspan( first_result, turn - first_result ), {} );
                    const dim_t added = static_cast<dim_t>( agent.model().tokenizer().encode( after.text ).size() );

                    const auto& prefill = reuse.rounds()[ first_round + static_cast<std::size_t>( round ) ];

                    ASSERT_TRUE( prefill.has_value() ) << name << " round " << round + 1 << " prefilled its whole prompt";

                    expected_prompt += generated[ static_cast<std::size_t>( round - 1 ) ] + added;

                    EXPECT_EQ( prefill->prompt, expected_prompt )
                        << name << " round " << round + 1 << " is not the last round's prompt, its reply and its results";

                    const dim_t prefilled = prefill->prompt - prefill->skipped;

                    std::printf( "%s\n", std::format( "{}: message {} round {} prefilled {} of {} prompt tokens; the results add {}",
                        name, message + 1, round + 1, prefilled, prefill->prompt, added ).c_str() );

                    // The first token the results add is the token the reply stopped at, which the model holds
                    // when its reply ended on the marker the template opens the results with.
                    EXPECT_LE( prefilled, added ) << name << " round " << round + 1 << " prefilled more than its results";
                    EXPECT_GE( prefilled, added - 1 ) << name << " round " << round + 1;
                }
            }
        }

        /// A store root under the temp directory, removed on destruction.
        class ScratchStore
        {
        public:

            ScratchStore()
            {
                static int counter = 0;

                root_ = std::filesystem::temp_directory_path() / std::format( "mila_agent_test_{}", counter++ );

                std::error_code ignored;
                std::filesystem::remove_all( root_, ignored );
            }

            ~ScratchStore()
            {
                std::error_code ignored;
                std::filesystem::remove_all( root_, ignored );
            }

            ModelStore store() const { return ModelStore( root_ ); }

            /// Record a model whose weights and tokenizer are each a few bytes, enough to be complete.
            void install( std::string name, std::string architecture, bool instruct ) const
            {
                ModelStore target = store();

                ModelRecord record;
                record.name = std::move( name );
                record.architecture = std::move( architecture );
                record.capabilities.instruct = instruct;

                for ( const char* role : { "weights", "tokenizer" } )
                {
                    const std::string payload = record.name + role;
                    const std::string digest = sha256Hex( payload.data(), payload.size() );

                    std::filesystem::create_directories( target.blobPath( digest ).parent_path() );

                    {
                        std::FILE* file = std::fopen( target.blobPath( digest ).string().c_str(), "wb" );
                        std::fwrite( payload.data(), 1, payload.size(), file );
                        std::fclose( file );
                    }

                    ModelFile entry;
                    entry.role = role;
                    entry.path = std::string( role ) + ".bin";
                    entry.sha256 = digest;
                    entry.bytes = payload.size();

                    record.files.push_back( entry );
                }

                target.writeRecord( record );
            }

        private:

            std::filesystem::path root_;
        };
    }

    // ================================================================
    // The gate: one program, three families, only the name changed
    // ================================================================

    TEST( AgentCuda, GemmaStreamsAndCallsATool )
    {
        // The 26B-A4B first: Mila measured it as an agent, so the gate also asks it to answer from the result.
        const auto name = firstInstalled( { "gemma-4-26B-A4B-it-q4_0", "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        askForTheWeather<GemmaModel<DeviceType::Cuda, TensorDataType::BF16>>( *name );
    }

    TEST( AgentCuda, LlamaStreamsAndCallsATool )
    {
        // The small 3.2 models are not agents, so the gate's Llama is 3.1 8B.
        const auto name = firstInstalled( { "Llama-3.1-8B-Instruct-fp4", "Llama-3.1-8B-Instruct-fp8" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        askForTheWeather<LlamaModel<DeviceType::Cuda, TensorDataType::BF16>>( *name );
    }

    TEST( AgentCuda, QwenStreamsAndCallsATool )
    {
        const auto name = firstInstalled( { "Qwen3.8-27B-fp4", "Qwen3.8-27B-fp4-cb3" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        askForTheWeather<QwenModel<DeviceType::Cuda, TensorDataType::BF16>>( *name );
    }

    // ================================================================
    // A round prefills only what it adds (ToolCalling.md section 4)
    // ================================================================

    TEST( AgentCuda, GemmaPrefillsOnlyWhatEachRoundAdds )
    {
        const auto name = firstInstalled( { "gemma-4-26B-A4B-it-q4_0", "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        expectEachRoundPrefillsOnlyWhatItAdds( *name );
    }

    TEST( AgentCuda, LlamaPrefillsOnlyWhatEachRoundAdds )
    {
        const auto name = firstInstalled( { "Llama-3.1-8B-Instruct-fp4", "Llama-3.1-8B-Instruct-fp8" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        expectEachRoundPrefillsOnlyWhatItAdds( *name );
    }

    TEST( AgentCuda, QwenPrefillsOnlyWhatEachRoundAdds )
    {
        const auto name = firstInstalled( { "Qwen3.8-27B-fp4", "Qwen3.8-27B-fp4-cb3" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        expectEachRoundPrefillsOnlyWhatItAdds( *name );
    }

    // ================================================================
    // Every call passes the question: one Gemma agent, a test per outcome
    // ================================================================

    class AgentOnGemmaCuda : public ::testing::Test
    {
    protected:

        static void SetUpTestSuite()
        {
            const auto name = firstInstalled( { "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } );

            if ( whyNotRunnable( name ).has_value() )
            {
                return;
            }

            agent_ = std::make_unique<Agent>( Agent::create( *name, *requestThatFits( *name ), Session{}
                .withTool( Tool{
                    "get_weather",
                    "Get the current weather in a city.",
                    kWeatherParameters,
                    []( std::string_view arguments, const ToolGrant&, std::stop_token stop )
                    {
                        arguments_seen_.emplace_back( arguments );

                        if ( tool_throws_ )
                        {
                            throw std::runtime_error( "the weather service is down" );
                        }

                        if ( tool_body_ )
                        {
                            return tool_body_( stop );
                        }

                        return std::string( kWeatherResult );
                    },
                    "Agent tests",
                    "Looks up the weather." } ) ) );
        }

        static void TearDownTestSuite()
        {
            agent_.reset();
        }

        void SetUp() override
        {
            if ( !agent_ )
            {
                GTEST_SKIP() << "no Gemma 4 12B is installed, or no CUDA device holds it";
            }

            agent_->session().newConversation();
            agent_->session().onAsk( {} );
            agent_->session().withSystemPrompt( {} );
            agent_->session().withLimits( {} );
            agent_->session().withRules( {} );
            agent_->session().withSampling( {} );
            agent_->model().model().seedSampler( 7 );

            arguments_seen_.clear();
            tool_throws_ = false;
            tool_body_ = {};
        }

        /// Ask about the weather and collect the result of every call answered.
        std::vector<AgentEvent> respond( std::string_view message, std::stop_token stop = {} )
        {
            std::vector<AgentEvent> called;

            last_ = agent_->respond( message,
                [&called, this]( const AgentEvent& event )
                {
                    switch ( event.kind )
                    {
                        case AgentEvent::Kind::ToolCalled:
                            results_.emplace_back( event.text );
                            called.push_back( event );
                            break;

                        case AgentEvent::Kind::RoundBegan:
                            budgets_.push_back( event.budget );
                            tokens_.emplace_back();
                            break;

                        case AgentEvent::Kind::Token:
                            tokens_.back().emplace_back( event.text );
                            break;

                        default:
                            break;
                    }
                },
                stop );

            return called;
        }

        static void allowEveryCall()
        {
            agent_->session().onAsk( []( const ToolQuestion& ) { return ToolAnswer::Once; } );
        }

        static inline std::unique_ptr<Agent> agent_;
        static inline std::vector<std::string> arguments_seen_;
        static inline bool tool_throws_ = false;

        /// What the tool does in place of reporting the weather, when set.
        static inline std::function<std::string( std::stop_token )> tool_body_;

        Response last_;
        std::vector<std::string> results_;

        /// Each round's budget, and each round's tokens as decoded.
        std::vector<std::optional<dim_t>> budgets_;
        std::vector<std::vector<std::string>> tokens_;
    };

    TEST_F( AgentOnGemmaCuda, WithNoOneToAskACallIsDeclinedAndTheModelReadsSo )
    {
        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << "Gemma answered without calling the tool: " << last_.text;
        EXPECT_TRUE( arguments_seen_.empty() ) << "a call ran with no one to ask";
        EXPECT_TRUE( called.front().is_error );
        EXPECT_EQ( results_.front(), "The user declined to run this tool call." );

        const auto conversation = agent_->session().conversation();
        EXPECT_EQ( conversation[ 2 ].content, "The user declined to run this tool call." );
        EXPECT_EQ( conversation.back().role, Conversation::Role::Assistant );
    }

    TEST_F( AgentOnGemmaCuda, ADeclinedCallDoesNotRun )
    {
        int asked = 0;

        agent_->session().onAsk( [&asked]( const ToolQuestion& question )
            {
                ++asked;
                EXPECT_EQ( question.name, "get_weather" );
                EXPECT_EQ( question.summary, "Looks up the weather." );
                EXPECT_EQ( question.source, "Agent tests" );
                EXPECT_NE( question.arguments.find( "Paris" ), std::string_view::npos ) << question.arguments;

                return ToolAnswer::Deny;
            } );

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << last_.text;
        EXPECT_GE( asked, 1 );
        EXPECT_TRUE( arguments_seen_.empty() );
        EXPECT_TRUE( called.front().is_error );
        EXPECT_EQ( called.front().decision, ToolDecision::DeclinedByPerson );
    }

    TEST_F( AgentOnGemmaCuda, ACallTheRulesAllowRunsWithoutAskingAndIsRecorded )
    {
        int asked = 0;

        agent_->session().onAsk( [&asked]( const ToolQuestion& )
            {
                ++asked;

                return ToolAnswer::Deny;
            } );
        agent_->session().withRules( ToolRules{}.add( { "get_weather", ToolPermission::Allow } ) );

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << last_.text;
        EXPECT_EQ( asked, 0 );
        EXPECT_FALSE( arguments_seen_.empty() );
        EXPECT_EQ( called.front().decision, ToolDecision::AllowedByRule );
        EXPECT_EQ( called.front().returned_bytes, std::string_view( kWeatherResult ).size() );
        EXPECT_FALSE( called.front().is_error );
    }

    TEST_F( AgentOnGemmaCuda, ACallTheRulesDenyDoesNotRunAndTheModelReadsSo )
    {
        allowEveryCall();
        agent_->session().withRules( ToolRules{}.add( { "*", ToolPermission::Deny } ) );

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << last_.text;
        EXPECT_TRUE( arguments_seen_.empty() );
        EXPECT_EQ( called.front().decision, ToolDecision::DeniedByRule );
        EXPECT_EQ( agent_->session().conversation()[ 2 ].content, "The user's rules do not allow this tool call." );
        EXPECT_EQ( agent_->session().conversation().back().role, Conversation::Role::Assistant );
    }

    TEST_F( AgentOnGemmaCuda, AToolAllowedForTheConversationIsNotAskedAboutAgainUntilItEnds )
    {
        int asked = 0;

        agent_->session().onAsk( [&asked]( const ToolQuestion& )
            {
                ++asked;

                return ToolAnswer::ForConversation;
            } );

        ASSERT_FALSE( respond( "What is the weather in Paris right now?" ).empty() ) << last_.text;
        ASSERT_FALSE( respond( "And what is the weather in Rome right now?" ).empty() ) << last_.text;

        EXPECT_EQ( asked, 1 );
        EXPECT_GE( arguments_seen_.size(), 2u );

        agent_->session().newConversation();

        ASSERT_FALSE( respond( "What is the weather in Oslo right now?" ).empty() ) << last_.text;
        EXPECT_EQ( asked, 2 ) << "an answer for one conversation was kept into the next";
    }

    TEST_F( AgentOnGemmaCuda, AToolThatThrowsReachesTheModelAsAnErrorResult )
    {
        agent_->session().onAsk( []( const ToolQuestion& ) { return ToolAnswer::Once; } );
        tool_throws_ = true;

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << last_.text;
        EXPECT_TRUE( called.front().is_error );
        EXPECT_NE( results_.front().find( "the weather service is down" ), std::string::npos ) << results_.front();
        EXPECT_EQ( agent_->session().conversation().back().role, Conversation::Role::Assistant );
    }

    TEST_F( AgentOnGemmaCuda, AReplyStoppedWhileACallWaitsRunsNothingAndEndsTheResponse )
    {
        std::stop_source stop;

        agent_->session().onAsk( [&stop]( const ToolQuestion& )
            {
                stop.request_stop();

                return ToolAnswer::Once;
            } );

        const auto called = respond( "What is the weather in Paris right now?", stop.get_token() );

        EXPECT_TRUE( called.empty() );
        EXPECT_TRUE( arguments_seen_.empty() ) << "a call ran after the reply was stopped";
        EXPECT_EQ( last_.status, GenerateStatus::ClientCancelled );
        EXPECT_EQ( last_.rounds, 1 );

        const auto conversation = agent_->session().conversation();
        ASSERT_EQ( conversation.size(), 3u );
        EXPECT_EQ( conversation.back().role, Conversation::Role::Tool );
        EXPECT_EQ( conversation.back().content, "The user stopped the reply before this tool call ran." );
    }

    TEST_F( AgentOnGemmaCuda, AMessageTheContextCannotHoldIsNotGeneratedFromAndDoesNotJoin )
    {
        std::string long_prompt;

        for ( int word = 0; word < 12000; ++word )
        {
            long_prompt += "weather ";
        }

        agent_->session().withSystemPrompt( long_prompt );

        const auto called = respond( "What is the weather in Paris right now?" );

        EXPECT_TRUE( called.empty() );
        EXPECT_EQ( last_.status, GenerateStatus::ContextOverflow );
        EXPECT_EQ( last_.rounds, 0 );
        EXPECT_TRUE( agent_->session().conversation().empty() );
    }

    TEST_F( AgentOnGemmaCuda, RefusesAToolOrSettingItCannotUse )
    {
        const auto run = []( std::string_view, const ToolGrant&, std::stop_token ) { return std::string(); };

        EXPECT_THROW( agent_->session().withTool( Tool{ "get_weather", "again", kWeatherParameters, run } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withTool( Tool{ "", "nameless", kWeatherParameters, run } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withTool( Tool{ "no_function", "", kWeatherParameters, {} } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withTool( Tool{ "bad_schema", "", "{not json", run } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withTool( Tool{ "array_schema", "", "[]", run } ), std::invalid_argument );

        EXPECT_THROW( agent_->withTemplateVariable( "no_such_variable", "true" ), std::invalid_argument );
        EXPECT_THROW( agent_->withTemplateVariable( "enable_thinking", "maybe" ), std::invalid_argument );
        EXPECT_THROW( agent_->withTemplateVariables( { { "enable_thinking", "maybe" } } ), std::invalid_argument );

        EXPECT_THROW( agent_->session().withLimits( { .rounds = 0 } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withLimits( { .time = std::chrono::milliseconds( 0 ) } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withLimits( { .reply_tokens = 0 } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withLimits( { .tool_time = std::chrono::milliseconds( 0 ) } ), std::invalid_argument );
        EXPECT_THROW( agent_->session().withLimits( { .result_tokens = 0 } ), std::invalid_argument );
    }

    // ================================================================
    // A response runs within its limits (ToolCalling.md section 4)
    // ================================================================

    TEST_F( AgentOnGemmaCuda, TheLastRoundAllowedAnswersItsCallsWithoutRunningThem )
    {
        allowEveryCall();
        agent_->session().withLimits( { .rounds = 1 } );

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << "Gemma answered without calling the tool: " << last_.text;
        EXPECT_TRUE( arguments_seen_.empty() ) << "a call of the last round ran";
        EXPECT_TRUE( called.front().is_error );
        EXPECT_EQ( results_.front(), "This tool call was not run: the response reached its limit of 1 rounds." );
        EXPECT_TRUE( last_.round_limit_reached );
        EXPECT_EQ( last_.rounds, 1 );

        // The calls keep their results, so the conversation renders and the next response can ask again.
        const auto conversation = agent_->session().conversation();
        ASSERT_EQ( conversation.size(), 3u );
        EXPECT_EQ( conversation.back().role, Conversation::Role::Tool );
        EXPECT_NO_THROW( (void)agent_->prompt( "And now?" ) );
    }

    TEST_F( AgentOnGemmaCuda, ReplyTokensBoundTheWholeResponseAcrossItsRounds )
    {
        allowEveryCall();
        agent_->session().withLimits( { .reply_tokens = 4000 } );

        ASSERT_FALSE( respond( "What is the weather in Paris right now?" ).empty() ) << last_.text;
        ASSERT_GE( budgets_.size(), 2u );

        // The first round may take every token; the next only what the first left.
        EXPECT_EQ( budgets_[ 0 ], dim_t( 4000 ) );
        EXPECT_EQ( budgets_[ 1 ], dim_t( 4000 ) - static_cast<dim_t>( tokens_[ 0 ].size() ) );
    }

    TEST_F( AgentOnGemmaCuda, AResponseOutOfReplyTokensStopsAndRunsNothing )
    {
        allowEveryCall();
        agent_->session().withLimits( { .reply_tokens = 3 } );

        const auto called = respond( "What is the weather in Paris right now?" );

        EXPECT_TRUE( called.empty() );
        EXPECT_TRUE( arguments_seen_.empty() );
        EXPECT_EQ( last_.status, GenerateStatus::MaxNewTokensReached ) << to_string( last_.status );
        ASSERT_EQ( tokens_.size(), 1u );
        EXPECT_EQ( tokens_[ 0 ].size(), 3u );
        EXPECT_EQ( agent_->session().conversation().back().role, Conversation::Role::Assistant );
    }

    TEST_F( AgentOnGemmaCuda, TokenEventsAreTheModelsOutputVerbatim )
    {
        allowEveryCall();

        ASSERT_FALSE( respond( "What is the weather in Paris right now?" ).empty() ) << last_.text;
        ASSERT_GE( tokens_.size(), 2u );

        std::string first_round;
        std::string last_round;

        for ( const auto& piece : tokens_.front() )
        {
            first_round += piece;
        }

        for ( const auto& piece : tokens_.back() )
        {
            last_round += piece;
        }

        // The call as the template writes it, markers and all, which no Answer event carries.
        EXPECT_NE( first_round.find( "<|tool_call>" ), std::string::npos ) << first_round;

        // And the answer the last round streamed, read from those same tokens.
        const std::size_t end = last_.text.find_last_not_of( " \n" );

        ASSERT_NE( end, std::string::npos ) << "the response has no text";
        ASSERT_GE( end, 12u );
        EXPECT_NE( last_round.find( last_.text.substr( end - 12, 13 ) ), std::string::npos ) << last_round;
    }

    TEST_F( AgentOnGemmaCuda, AResultLargerThanTheContextLeftIsCutAndTheModelToldSo )
    {
        allowEveryCall();

        std::string forecast;

        for ( int day = 0; forecast.size() < 64'000; ++day )
        {
            forecast += std::format( "Day {}: sunny, 17 degrees Celsius, light wind from the west. ", day );
        }

        tool_body_ = [&forecast]( std::stop_token ) { return forecast; };

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << last_.text;
        EXPECT_FALSE( called.front().is_error ) << results_.front();

        const std::string& result = results_.front();
        const std::size_t note = result.rfind( "\n[The result was cut to its first " );

        ASSERT_NE( note, std::string::npos ) << result.substr( result.size() - 200 );
        EXPECT_TRUE( forecast.starts_with( result.substr( 0, note ) ) ) << "the cut is not the result's beginning";
        EXPECT_EQ( agent_->session().conversation()[ 2 ].content, result );

        // The cut leaves the model a position to answer in, so the response never overflowed with no word of why.
        EXPECT_EQ( last_.rounds, 2 );
    }

    TEST_F( AgentOnGemmaCuda, AResultLongerThanAToolMayGiveIsCutAndTheModelToldSo )
    {
        allowEveryCall();
        agent_->session().withLimits( { .result_tokens = 40 } );

        std::string forecast;

        for ( int day = 0; day < 30; ++day )
        {
            forecast += std::format( "Day {}: sunny, 17 degrees Celsius, light wind from the west. ", day );
        }

        tool_body_ = [&forecast]( std::stop_token ) { return forecast; };

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << last_.text;
        EXPECT_FALSE( called.front().is_error ) << results_.front();
        EXPECT_EQ( called.front().returned_bytes, forecast.size() );

        const std::string& result = results_.front();
        const std::size_t note = result.rfind( "\n[The result was cut to its first 40 of " );

        ASSERT_NE( note, std::string::npos ) << result;
        EXPECT_TRUE( result.ends_with( " tokens, the most this tool may give.]" ) ) << result;
        EXPECT_TRUE( forecast.starts_with( result.substr( 0, note ) ) ) << "the cut is not the result's beginning";
        EXPECT_EQ( agent_->session().conversation()[ 2 ].content, result );
    }

    TEST_F( AgentOnGemmaCuda, ACallOutOfItsTimeEndsAndTheResponseGoesOn )
    {
        allowEveryCall();
        agent_->session().withLimits( { .rounds = 2, .tool_time = std::chrono::milliseconds( 200 ) } );

        bool tool_saw_stop = false;

        tool_body_ = [&tool_saw_stop]( std::stop_token running )
            {
                std::mutex mutex;
                std::condition_variable_any never;
                std::unique_lock<std::mutex> lock( mutex );

                never.wait_for( lock, running, std::chrono::minutes( 1 ), [] { return false; } );
                tool_saw_stop = running.stop_requested();

                return std::string( kWeatherResult );
            };

        const auto called = respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( called.empty() ) << last_.text;
        EXPECT_TRUE( tool_saw_stop );
        EXPECT_TRUE( called.front().is_error );
        EXPECT_EQ( results_.front(), "Error: tool 'get_weather' ran out of its 200 ms, so it may not have finished." );
        EXPECT_NE( last_.status, GenerateStatus::ClientCancelled ) << "a call out of its time ended the response";
        EXPECT_EQ( last_.rounds, 2 );
    }

    TEST_F( AgentOnGemmaCuda, AStopWhileACallRunsAsksTheCallToStopAndEndsTheResponse )
    {
        allowEveryCall();

        std::stop_source stop;
        bool tool_saw_stop = false;

        tool_body_ = [&stop, &tool_saw_stop]( std::stop_token running )
            {
                stop.request_stop();
                tool_saw_stop = running.stop_requested();

                return std::string( "partial" );
            };

        respond( "What is the weather in Paris right now?", stop.get_token() );

        ASSERT_FALSE( arguments_seen_.empty() ) << last_.text;
        EXPECT_TRUE( tool_saw_stop ) << "the running call was not asked to stop";
        EXPECT_TRUE( results_.empty() ) << "a stopped call was reported";
        EXPECT_EQ( last_.status, GenerateStatus::ClientCancelled );
        EXPECT_FALSE( last_.time_limit_reached );
        EXPECT_EQ( agent_->session().conversation().back().content,
            "The user stopped the reply while this tool call ran, so it may not have finished." );
    }

    TEST_F( AgentOnGemmaCuda, ATimeLimitEndsARunningCallAndTheResponse )
    {
        allowEveryCall();
        agent_->session().withLimits( { .time = std::chrono::seconds( 10 ) } );

        bool tool_saw_stop = false;

        tool_body_ = [&tool_saw_stop]( std::stop_token running )
            {
                std::mutex mutex;
                std::condition_variable_any never;
                std::unique_lock<std::mutex> lock( mutex );

                // Only a stop ends the wait, and a minute is far past the limit.
                never.wait_for( lock, running, std::chrono::minutes( 1 ), [] { return false; } );
                tool_saw_stop = running.stop_requested();

                return std::string( kWeatherResult );
            };

        respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( arguments_seen_.empty() ) << last_.text;
        EXPECT_TRUE( tool_saw_stop );
        EXPECT_TRUE( last_.time_limit_reached );
        EXPECT_EQ( last_.status, GenerateStatus::ClientCancelled );
        EXPECT_EQ( agent_->session().conversation().back().content,
            "The response ran out of time while this tool call ran, so it may not have finished." );
    }

    TEST_F( AgentOnGemmaCuda, ThePersonsTimeToAnswerIsNotTheResponses )
    {
        agent_->session().withLimits( { .time = std::chrono::seconds( 8 ) } );
        agent_->session().onAsk( []( const ToolQuestion& )
            {
                std::this_thread::sleep_for( std::chrono::seconds( 9 ) );

                return ToolAnswer::Once;
            } );

        respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( arguments_seen_.empty() ) << last_.text;
        EXPECT_FALSE( last_.time_limit_reached );
        EXPECT_EQ( last_.status, GenerateStatus::Success ) << to_string( last_.status );
        EXPECT_EQ( agent_->session().conversation()[ 2 ].content, kWeatherResult );
    }

    TEST_F( AgentOnGemmaCuda, ThePromptOfANextMessageLeavesTheConversationAlone )
    {
        agent_->session().withSystemPrompt( "You are terse." );

        const Conversation::Prompt prompt = agent_->prompt( "Is it raining in Lima?" );

        EXPECT_NE( prompt.text.find( "You are terse." ), std::string::npos );
        EXPECT_NE( prompt.text.find( "Is it raining in Lima?" ), std::string::npos );
        EXPECT_NE( prompt.text.find( "get_weather" ), std::string::npos ) << "the tools are not in the prompt";
        EXPECT_TRUE( agent_->session().conversation().empty() );
    }

    // ================================================================
    // A base model is refused before anything is read
    // ================================================================

    TEST( Agent, RefusesABaseModelBeforeReadingItsFiles )
    {
        // The files are a few bytes each: reading either would fail differently from the refusal.
        ScratchStore scratch;
        scratch.install( "gemma-base", "gemma", false );

        const std::optional<OpenRefusal> checked = Agent::check( "gemma-base", requestOn( 0 ), scratch.store() );

        ASSERT_TRUE( checked.has_value() );
        EXPECT_EQ( checked->reason, OpenRefusal::Reason::BaseModel );

        try
        {
            (void)Agent::create( "gemma-base", requestOn( 0 ), {}, scratch.store() );
            ADD_FAILURE() << "a base model opened as an agent";
        }
        catch ( const OpenRefusedError& error )
        {
            EXPECT_EQ( error.refusal().reason, OpenRefusal::Reason::BaseModel );
            EXPECT_NE( std::string( error.what() ).find( "base model" ), std::string::npos ) << error.what();
        }
    }

    TEST( Agent, ChecksAnInstructModelThroughItsHandleAndOpensNothing )
    {
        ScratchStore scratch;
        scratch.install( "gemma-chat", "gemma", true );

        // Its weights are a few bytes, so a check that read them would fail; one that reads only the record passes.
        EXPECT_FALSE( Agent::check( "gemma-chat", requestOn( 0 ), scratch.store() ).has_value() );

        const std::optional<OpenRefusal> absent = Agent::check( "not-installed", requestOn( 0 ), scratch.store() );

        ASSERT_TRUE( absent.has_value() );
        EXPECT_EQ( absent->reason, OpenRefusal::Reason::NotInstalled );
        EXPECT_EQ( absent->name, "not-installed" );
    }

    TEST( Agent, ACreateThatFailsLeavesTheSessionAsItWas )
    {
        ScratchStore scratch;
        Session session;
        std::vector<std::string> arguments_seen;

        session.withTool( weatherTool( arguments_seen ) ).withSystemPrompt( "You are terse." );
        session.openConversation( { { Conversation::Role::User, "Hello" }, { Conversation::Role::Assistant, "Hi." } } );

        EXPECT_THROW( (void)Agent::create( "not-installed", requestOn( 0 ), std::move( session ), scratch.store() ),
            OpenRefusedError );

        // Moved from only once a model opens.
        EXPECT_EQ( session.conversation().size(), 2u );
        EXPECT_EQ( session.tools().size(), 1u );
        EXPECT_EQ( session.systemPrompt(), "You are terse." );
    }

    // ================================================================
    // A session outlives its model (Direction.md 4.3)
    // ================================================================

    TEST( AgentCuda, ALoadTheDeviceRefusesLeavesTheSessionAsItWas )
    {
        const auto name = firstInstalled( { "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        Session session;
        session.openConversation( { { Conversation::Role::User, "Hello" }, { Conversation::Role::Assistant, "Hi." } } );

        // Past any context the model addresses: refused by the planner, after the check passed.
        DeploymentRequest request = *requestThatFits( *name );
        request.withContextLength( dim_t( 1 ) << 30 );

        ASSERT_FALSE( Agent::check( *name, request ).has_value() );
        EXPECT_ANY_THROW( (void)Agent::create( *name, request, std::move( session ) ) );
        EXPECT_EQ( session.conversation().size(), 2u );
    }

    TEST( AgentCuda, AReloadAtAnotherContextKeepsTheConversationAndItsTools )
    {
        const auto name = firstInstalled( { "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        std::vector<std::string> arguments_seen;
        DeploymentRequest request = *requestThatFits( *name );

        auto agent = Agent::create( *name, request, Session{}
            .withTool( weatherTool( arguments_seen ) )
            .withRules( ToolRules{}.add( { "get_weather", ToolPermission::Allow } ) ) );

        agent.model().model().seedSampler( 7 );

        const Response first = agent.respond( "What is the weather in Paris right now?" );

        ASSERT_FALSE( arguments_seen.empty() ) << "Gemma answered without calling the tool: " << first.text;

        const std::size_t turns = agent.session().conversation().size();

        request.withContextLength( kContextLength / 2 );

        Session held = std::move( agent ).release();
        agent = Agent::create( *name, request, std::move( held ) );

        EXPECT_EQ( agent.plan().contextLength(), kContextLength / 2 );
        ASSERT_EQ( agent.session().conversation().size(), turns );
        EXPECT_EQ( agent.session().tools().size(), 1u );

        const Response next = agent.respond( "And what is the weather in Rome right now?" );

        EXPECT_EQ( next.status, GenerateStatus::Success ) << to_string( next.status );
        EXPECT_GT( agent.session().conversation().size(), turns );
    }

    TEST( AgentCuda, OneSessionCarriesItsToolsRulesAndConversationThroughEveryFamily )
    {
        // One model resident at a time, each released before the next is planned.
        const std::vector<std::optional<std::string>> names{
            firstInstalled( { "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } ),
            firstInstalled( { "Qwen3.8-27B-fp4", "Qwen3.8-27B-fp4-cb3" } ),
            firstInstalled( { "Llama-3.1-8B-Instruct-fp4", "Llama-3.1-8B-Instruct-fp8" } ) };

        std::vector<std::string> runnable;

        for ( const auto& name : names )
        {
            if ( !whyNotRunnable( name ) )
            {
                runnable.push_back( *name );
            }
        }

        if ( runnable.size() < 2 )
        {
            GTEST_SKIP() << "fewer than two families can run here";
        }

        std::vector<std::string> arguments_seen;
        int asked = 0;

        Session session;
        session.withTool( weatherTool( arguments_seen ) )
            .withRules( ToolRules{}.add( { "get_weather", ToolPermission::Allow } ) )
            .onAsk( [&asked]( const ToolQuestion& )
                {
                    ++asked;

                    return ToolAnswer::Deny;
                } )
            .withSystemPrompt( "Answer in one sentence." );

        const std::vector<std::string> messages{
            "What is the weather in Paris right now?", "And in Rome?", "Which of the two cities is warmer?" };

        std::size_t turns = 0;

        for ( std::size_t index = 0; index < runnable.size(); ++index )
        {
            auto agent = Agent::create( runnable[ index ], *requestThatFits( runnable[ index ] ), std::move( session ) );

            agent.model().model().seedSampler( 7 );

            // What the last model left, in this model's own template.
            ASSERT_EQ( agent.session().conversation().size(), turns ) << runnable[ index ];

            const Conversation::Prompt prompt = agent.prompt();

            EXPECT_NE( prompt.text.find( "get_weather" ), std::string::npos ) << runnable[ index ] << ": the tools did not cross";
            EXPECT_NE( prompt.text.find( "Answer in one sentence." ), std::string::npos ) << runnable[ index ];

            if ( index > 0 )
            {
                EXPECT_NE( prompt.text.find( messages.front() ), std::string::npos )
                    << runnable[ index ] << ": the conversation did not cross";
            }

            const Response response = agent.respond( messages[ index ] );

            EXPECT_EQ( response.status, GenerateStatus::Success ) << runnable[ index ] << ": " << to_string( response.status );
            EXPECT_GT( agent.session().conversation().size(), turns ) << runnable[ index ];

            turns = agent.session().conversation().size();
            session = std::move( agent ).release();
        }

        // Every call the models made was allowed by the rule, and none was asked about.
        EXPECT_EQ( asked, 0 );
        EXPECT_FALSE( arguments_seen.empty() ) << "no model called the tool";
    }

    TEST( AgentCuda, ATemplateVariableTheNextModelDoesNotAcceptIsDroppedAndNamed )
    {
        const auto name = firstInstalled( { "gemma-4-12b-it-fp4", "gemma-4-12b-it-q4_0" } );

        if ( const auto skipped = whyNotRunnable( name ) )
        {
            GTEST_SKIP() << *skipped;
        }

        // Llama's own variable, which Gemma's template does not have; Gemma's thinking switch, which it does.
        auto agent = Agent::create( *name, *requestThatFits( *name ), Session{}
            .withTemplateVariable( "tools_in_user_message", "false" )
            .withTemplateVariable( "enable_thinking", "false" ) );

        ASSERT_EQ( agent.droppedTemplateVariables().size(), 1u );
        EXPECT_EQ( agent.droppedTemplateVariables().front(), "tools_in_user_message" );
        EXPECT_FALSE( agent.session().templateVariables().contains( "tools_in_user_message" ) );
        EXPECT_EQ( agent.session().templateVariables().at( "enable_thinking" ), "false" );

        // Set on the session of a live agent, an unaccepted one is refused before the message joins.
        agent.session().withTemplateVariable( "no_such_variable", "true" );

        EXPECT_THROW( (void)agent.respond( "Hello" ), std::invalid_argument );
        EXPECT_TRUE( agent.session().conversation().empty() );
    }

    TEST_F( AgentOnGemmaCuda, ASamplingSettingTheSessionNamesIsTheOneUsed )
    {
        // Greedy whatever the seed: the publisher's sampling for Gemma is not, so two seeds would part.
        agent_->session().withSampling( SamplingOverride{ .top_k = 1 } ).withLimits( { .reply_tokens = 48 } );

        std::vector<std::string> replies;

        for ( const std::uint64_t seed : { 7u, 8u } )
        {
            agent_->session().newConversation();
            agent_->model().model().seedSampler( seed );

            replies.push_back( agent_->respond( "Name three colours." ).text );
        }

        ASSERT_FALSE( replies.front().empty() );
        EXPECT_EQ( replies.front(), replies.back() );
    }
}
