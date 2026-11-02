// The chat template a model runs with, from the architecture its manifest declares
// (Dnn.Models.ChatProtocolForArchitecture).

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    namespace Conversation = Mila::Dnn::Conversation;

    using Mila::Dnn::chatProtocolFor;

    namespace
    {
        std::map<std::string, std::string> variableDefaults( const Conversation::Protocol& protocol )
        {
            std::map<std::string, std::string> defaults;

            for ( const auto& variable : protocol.templateVariables() )
            {
                defaults[ variable.name ] = variable.default_value;
            }

            return defaults;
        }

        std::string renderPlain( const Conversation::Protocol& protocol )
        {
            const std::vector<Conversation::Turn> history{ { Conversation::Role::User, "hi" } };

            return protocol.formatPrompt( history, {} ).text;
        }
    }

    TEST( ChatProtocolForArchitecture, AFamilyWithOneTemplateTakesAnUndeclaredVersionAsIt )
    {
        // Every Gemma and Qwen manifest published before architecture_version existed.
        EXPECT_TRUE( renderPlain( *chatProtocolFor( "gemma", "" ) ).starts_with( "<bos><|turn>user\n" ) );
        EXPECT_EQ( variableDefaults( *chatProtocolFor( "qwen", "" ) ).at( "reasoning_effort" ), "xhigh" );
    }

    TEST( ChatProtocolForArchitecture, LlamaIsChosenByItsDeclaredVersion )
    {
        EXPECT_EQ( variableDefaults( *chatProtocolFor( "llama", "3.1" ) ).at( "date_string" ), "26 Jul 2024" );

        // 3.2's template dates the prompt today.
        EXPECT_EQ( variableDefaults( *chatProtocolFor( "llama", "3.2" ) ).at( "date_string" ), "" );
    }

    TEST( ChatProtocolForArchitecture, LlamaWithNoDeclaredVersionIsRefused )
    {
        // Its two templates differ; a guess would render one of them wrong.
        EXPECT_THROW( (void)chatProtocolFor( "llama", "" ), std::invalid_argument );
    }

    TEST( ChatProtocolForArchitecture, AnArchitectureWithNoTemplateIsRefused )
    {
        EXPECT_THROW( (void)chatProtocolFor( "gpt2", "" ), std::invalid_argument );
        EXPECT_THROW( (void)chatProtocolFor( "llama", "3.3" ), std::invalid_argument );
        EXPECT_THROW( (void)chatProtocolFor( "qwen", "4" ), std::invalid_argument );
    }
}
