/**
 * @file AllowedDevices.Cpu.cpp
 * @brief The devices a request allows, and the planner's choice among them, against devices the test describes.
 *
 * Specifications/Deployment.md 12.8 and section 5 across devices. The per-device plans are made up here, so
 * nothing reads or builds on a device and every case runs in any build.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <expected>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace
    {
        DeviceReading describedCuda( int index, std::string identity )
        {
            DeviceReading description{ .device = DeviceId{ DeviceType::Cuda, index } };
            description.hardware_name = "a card";
            description.stable_identity = std::move( identity );

            return description;
        }

        /// Two CUDA cards and the host, as DeviceReading::describePresent orders them.
        std::vector<DeviceReading> twoCardsAndTheHost()
        {
            DeviceReading host{ .device = DeviceId{ DeviceType::Cpu, 0 } };
            host.stable_identity = "cpu";

            return { host,
                describedCuda( 0, "GPU-aaaaaaaa-0000-0000-0000-000000000000" ),
                describedCuda( 1, "GPU-bbbbbbbb-0000-0000-0000-000000000000" ) };
        }

        DeploymentPlan planOn( DeviceId device, dim_t context_length, bool full_chunk )
        {
            DeviceReading reading{ .device = device };
            reading.free_bytes = 1;
            reading.total_bytes = 1;

            const PrefillChunking prefill{
                .chunk_rows = full_chunk ? 1024 : 256, .unconstrained_chunk_rows = 1024, .fits_available_memory = true };

            return DeploymentPlan( reading, context_length, DeploymentPlan::ContextLimit::DeviceMemory, prefill,
                MemoryStats{}, WeightQuantization::None, KvCacheCompression::None, Serialization::WeightsMetadata{}, "" );
        }

        std::unexpected<DeviceShortfall> shortfallOn( DeviceId device )
        {
            DeviceReading reading{ .device = device };

            return std::unexpected( DeviceShortfall( DeviceShortfall::Reason::WeightsExceedDevice, reading, 0, 1024, MemoryStats{} ) );
        }

        /// What each card answers, and which cards the planner asked.
        struct Answers
        {
            std::vector<std::expected<DeploymentPlan, DeviceShortfall>> per_card;
            std::vector<DeviceId> asked{};

            std::expected<DeploymentPlans, DeploymentRefusal> plan( const DeploymentRequest& request )
            {
                const std::vector<DeviceReading> present = twoCardsAndTheHost();

                return planOnAllowedDevices( request, DeviceType::Cuda, std::span<const DeviceReading>( present ),
                    [&]( DeviceId device ) -> std::expected<DeploymentPlan, DeviceShortfall>
                    {
                        asked.push_back( device );

                        return per_card.at( static_cast<std::size_t>( device.index ) );
                    } );
            }
        };
    }

    // ====================================================================
    // Reading the names a person writes
    // ====================================================================

    TEST( AllowedDevicesTests, NoNameAllowsEveryDevice )
    {
        const AllowedDevices every;

        EXPECT_TRUE( every.isEveryDevice() );

        for ( const DeviceReading& device : twoCardsAndTheHost() )
        {
            EXPECT_TRUE( every.allows( device ) ) << device.device.toString();
        }

        EXPECT_TRUE( every.absentFrom( twoCardsAndTheHost() ).empty() );
    }

    TEST( AllowedDevicesTests, ATypeAllowsEveryDeviceOfIt )
    {
        const AllowedDevices cuda = AllowedDevices::parse( "CUDA" );
        const std::vector<DeviceReading> present = twoCardsAndTheHost();

        EXPECT_FALSE( cuda.allows( present[ 0 ] ) );
        EXPECT_TRUE( cuda.allows( present[ 1 ] ) );
        EXPECT_TRUE( cuda.allows( present[ 2 ] ) );
    }

    TEST( AllowedDevicesTests, AnIndexAndAnIdentityEachAllowOneCard )
    {
        const std::vector<DeviceReading> present = twoCardsAndTheHost();

        const AllowedDevices by_index = AllowedDevices::parse( "cuda:1" );
        EXPECT_FALSE( by_index.allows( present[ 1 ] ) );
        EXPECT_TRUE( by_index.allows( present[ 2 ] ) );

        const AllowedDevices by_identity = AllowedDevices::parse( " gpu-AAAAAAAA-0000-0000-0000-000000000000 " );
        EXPECT_TRUE( by_identity.allows( present[ 1 ] ) ) << "an identity matches in any case, spaces trimmed";
        EXPECT_FALSE( by_identity.allows( present[ 2 ] ) );
    }

    TEST( AllowedDevicesTests, SeveralNamesAreOneSetAndEachAbsentNameIsKeptAsWritten )
    {
        const AllowedDevices allowed = AllowedDevices::parse( "cuda:1, GPU-ffffffff-0000-0000-0000-000000000000,cuda:7" );
        const std::vector<std::string> absent = allowed.absentFrom( twoCardsAndTheHost() );

        ASSERT_EQ( absent.size(), 2u );
        EXPECT_EQ( absent[ 0 ], "GPU-ffffffff-0000-0000-0000-000000000000" );
        EXPECT_EQ( absent[ 1 ], "cuda:7" );
        EXPECT_EQ( allowed.toString(), "cuda:1, GPU-ffffffff-0000-0000-0000-000000000000, cuda:7" );
    }

    TEST( AllowedDevicesTests, AnEmptyNameOrAnIndexThatDoesNotParseIsMalformed )
    {
        EXPECT_THROW( (void)AllowedDevices::parse( "" ), std::invalid_argument );
        EXPECT_THROW( (void)AllowedDevices::parse( "cuda:0,,cuda:1" ), std::invalid_argument );
        EXPECT_THROW( (void)AllowedDevices::parse( "cuda:one" ), std::invalid_argument );
        EXPECT_THROW( (void)AllowedDevices::parse( "card:0" ), std::invalid_argument );
    }

    // ====================================================================
    // The planner's choice among the allowed devices
    // ====================================================================

    TEST( AllowedDevicesTests, AFullPrefillChunkOutranksALongerContext )
    {
        Answers answers{ { planOn( Device::Cuda( 0 ), 65536, false ), planOn( Device::Cuda( 1 ), 32768, true ) } };
        const auto planned = answers.plan( DeploymentRequest{} );

        ASSERT_TRUE( planned.has_value() );
        ASSERT_EQ( planned->rankedPlans().size(), 2u );
        EXPECT_EQ( planned->best().device(), Device::Cuda( 1 ) );
        EXPECT_EQ( planned->rankedPlans()[ 1 ].device(), Device::Cuda( 0 ) );
    }

    TEST( AllowedDevicesTests, AtEqualChunksTheLongerContextWins )
    {
        Answers answers{ { planOn( Device::Cuda( 0 ), 32768, true ), planOn( Device::Cuda( 1 ), 65536, true ) } };
        const auto planned = answers.plan( DeploymentRequest{} );

        ASSERT_TRUE( planned.has_value() );
        EXPECT_EQ( planned->best().device(), Device::Cuda( 1 ) );
    }

    TEST( AllowedDevicesTests, PlansEqualOnBothRulesKeepDeviceOrder )
    {
        Answers answers{ { planOn( Device::Cuda( 0 ), 4096, true ), planOn( Device::Cuda( 1 ), 4096, true ) } };
        const auto planned = answers.plan( DeploymentRequest{} );

        ASSERT_TRUE( planned.has_value() );
        EXPECT_EQ( planned->best().device(), Device::Cuda( 0 ) ) << "nothing ranks devices by speed yet";
    }

    TEST( AllowedDevicesTests, OnlyTheAllowedCardsArePlannedAndACardThatDoesNotFitIsLeftOut )
    {
        Answers answers{ { planOn( Device::Cuda( 0 ), 65536, true ), planOn( Device::Cuda( 1 ), 4096, true ) } };
        const auto named = answers.plan( DeploymentRequest{}.withAllowedDevices( Device::Cuda( 1 ) ) );

        ASSERT_TRUE( named.has_value() );
        ASSERT_EQ( named->rankedPlans().size(), 1u );
        EXPECT_EQ( named->best().device(), Device::Cuda( 1 ) );
        EXPECT_EQ( answers.asked, std::vector<DeviceId>{ Device::Cuda( 1 ) } ) << "a card not allowed is never planned";

        Answers one_short{ { shortfallOn( Device::Cuda( 0 ) ), planOn( Device::Cuda( 1 ), 4096, true ) } };
        const auto either = one_short.plan( DeploymentRequest{} );

        ASSERT_TRUE( either.has_value() );
        ASSERT_EQ( either->rankedPlans().size(), 1u );
        EXPECT_EQ( either->best().device(), Device::Cuda( 1 ) );
    }

    // ====================================================================
    // Refusals
    // ====================================================================

    TEST( AllowedDevicesTests, ARefusalGivesEachAllowedCardsShortfallAndNamesTheCardsNotAllowed )
    {
        Answers answers{ { shortfallOn( Device::Cuda( 0 ) ), shortfallOn( Device::Cuda( 1 ) ) } };
        const auto refused = answers.plan( DeploymentRequest{}.withAllowedDevices( Device::Cuda( 0 ) ) );

        ASSERT_FALSE( refused.has_value() );
        EXPECT_EQ( refused.error().reason(), DeploymentRefusal::Reason::NoAllowedDeviceFits );
        ASSERT_EQ( refused.error().shortfalls().size(), 1u );
        EXPECT_EQ( refused.error().shortfalls().front().reading().device, Device::Cuda( 0 ) );
        ASSERT_EQ( refused.error().notAllowed().size(), 1u );
        EXPECT_EQ( refused.error().notAllowed().front().device, Device::Cuda( 1 ) )
            << "the host is not a device this model runs on, so it is not named";

        Answers every{ { shortfallOn( Device::Cuda( 0 ) ), shortfallOn( Device::Cuda( 1 ) ) } };
        const auto refused_everywhere = every.plan( DeploymentRequest{} );

        ASSERT_FALSE( refused_everywhere.has_value() );
        EXPECT_EQ( refused_everywhere.error().shortfalls().size(), 2u );
        EXPECT_TRUE( refused_everywhere.error().notAllowed().empty() );
    }

    TEST( AllowedDevicesTests, AnAbsentNameIsRefusedAndNothingIsPlanned )
    {
        Answers answers{ { planOn( Device::Cuda( 0 ), 4096, true ), planOn( Device::Cuda( 1 ), 4096, true ) } };
        const auto refused = answers.plan(
            DeploymentRequest{}.withAllowedDevices( AllowedDevices::parse( "cuda:0,GPU-ffffffff-0000-0000-0000-000000000000" ) ) );

        ASSERT_FALSE( refused.has_value() );
        EXPECT_EQ( refused.error().reason(), DeploymentRefusal::Reason::AllowedDeviceAbsent );
        ASSERT_EQ( refused.error().absent().size(), 1u );
        EXPECT_EQ( refused.error().absent().front(), "GPU-ffffffff-0000-0000-0000-000000000000" );
        EXPECT_TRUE( answers.asked.empty() ) << "a named card is never replaced by another";
    }

    TEST( AllowedDevicesTests, NoDeviceOfTheModelsTypeIsARefusal )
    {
        const std::vector<DeviceReading> host_only{ DeviceReading{ .device = DeviceId{ DeviceType::Cpu, 0 } } };
        const auto refused = planOnAllowedDevices( DeploymentRequest{}, DeviceType::Cuda,
            std::span<const DeviceReading>( host_only ),
            []( DeviceId device ) -> std::expected<DeploymentPlan, DeviceShortfall> { return planOn( device, 4096, true ); } );

        ASSERT_FALSE( refused.has_value() );
        EXPECT_EQ( refused.error().reason(), DeploymentRefusal::Reason::NoDeviceToPlanOn );
    }

    TEST( AllowedDevicesTests, AllowingAPresentDeviceOfAnotherTypeIsMalformed )
    {
        Answers answers{ { planOn( Device::Cuda( 0 ), 4096, true ), planOn( Device::Cuda( 1 ), 4096, true ) } };

        EXPECT_THROW( (void)answers.plan( DeploymentRequest{}.withAllowedDevices( AllowedDevices::parse( "cuda:0,cpu" ) ) ),
            std::invalid_argument );
    }
}
