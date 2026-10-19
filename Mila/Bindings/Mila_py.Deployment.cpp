/**
 * @file Mila_py.Deployment.cpp
 * @brief Implementation unit for Mila.Bindings.Deployment: asks the library's planner which weight formats fit.
 */

module Mila.Bindings.Deployment;

import std;
import Mila;

namespace Mila::Bindings
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;

    namespace
    {
        InstalledModel requireInstalled( const std::string& name )
        {
            const auto installed = InstalledModel::find( name );

            if ( !installed.has_value() )
            {
                throw std::runtime_error( std::format( "No model named '{}' is installed", name ) );
            }

            return *installed;
        }

        /// The format as a load names it, which is the word a user passes back.
        std::string formatName( WeightQuantization format )
        {
            switch ( format )
            {
                case WeightQuantization::FP8: return "fp8";
                case WeightQuantization::FP4: return "fp4";
                case WeightQuantization::Q4_0: return "q4_0";
                case WeightQuantization::Plan: return "cb2-3";
                case WeightQuantization::Fp4Codebook3: return "fp4-cb3";
                case WeightQuantization::None:
                default:
                    return "bf16";
            }
        }
    }

    std::string storedWeightQuantization( const std::string& name )
    {
        const InstalledModel model = requireInstalled( name );
        const std::optional<WeightQuantization> stored = model.storedWeightQuantization();

        if ( !stored.has_value() )
        {
            throw std::runtime_error( std::format(
                "'{}' is stored as '{}', which this build of Mila does not read",
                name, model.stored().record.weight_quantization ) );
        }

        return formatName( *stored );
    }

    std::optional<std::vector<std::string>> fittingWeightQuantizations( const std::string& name, int device_index )
    {
        const InstalledModel model = requireInstalled( name );

        DeploymentRequest request;
        request.withDevice( DeviceId{ DeviceType::Cuda, device_index } );
        request.withWholeDeviceMemory();

        const auto fitting = Mila::Deployment::fittingWeightQuantizations<DeviceType::Cuda>( model, request );

        if ( !fitting.has_value() )
        {
            return std::nullopt;
        }

        std::vector<std::string> names;

        for ( const WeightQuantization format : *fitting )
        {
            names.push_back( formatName( format ) );
        }

        return names;
    }
}
