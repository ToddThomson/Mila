/**
 * @file ModelHandle.cpp
 * @brief The bodies of ModelHandle's members: the device type and precision chosen at the open, and each family's
 *        load reached through its architecture. The interface is ModelHandle.ixx.
 */

module Deployment.ModelHandle;

import std;
import Compute.Device;
import Compute.DeviceRegistry;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.MemoryHold;
#ifdef MILA_HAS_CUDA
import Compute.DeviceTypeTraits.Cuda;
#endif
#ifdef MILA_HAS_METAL
import Compute.DeviceTypeTraits.Metal;
#endif
// Each family is instantiated here, so each backend's specializations must be visible from this file, as they are
// from a program that imports Mila: MSVC does not complete one that is only reachable (TensorOps.ixx).
import Compute.OperationTraits;
import Dnn.TensorOps;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.ILanguageModel;
import Dnn.PromptPrefixReuse;
import Dnn.Models.Conversation.Protocol;
import Dnn.Models.Architectures;
import Dnn.Models.ChatProtocolForArchitecture;
import Data.BpeTokenizer;
import Deployment.AllowedDevices;
import Deployment.DeviceReading;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlan;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Deployment.DeploymentRefusedError;
import Deployment.InstalledModel;
import Deployment.InstalledModelPlanner;
import Distribution.ModelCapabilities;
import Distribution.ModelStore;
import Dnn.WeightQuantization;
import Deployment.OpenRefusal;
import Deployment.OpenRefusedError;
import Serialization.WeightsReader;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Distribution;

    namespace
    {
        /// What only the typed model can answer, read where its type is known.
        struct LoadedModel
        {
            std::unique_ptr<ILanguageModel> model;
            std::optional<DeploymentPlan> plan;
            PromptPrefixReuse prefix_reuse{};
        };

        /**
         * @brief Run action at the device type given, for each device type this build carries.
         *
         * The one run-time to compile-time bridge for device types, as createExecutionContext is for contexts.
         */
        template<typename TAction>
        auto onDeviceType( DeviceType device_type, TAction&& action )
        {
            switch ( device_type )
            {
            #ifdef MILA_HAS_CUDA
                case DeviceType::Cuda:
                    return action.template operator()<DeviceType::Cuda>();
            #endif

            #ifdef MILA_HAS_METAL
                case DeviceType::Metal:
                    return action.template operator()<DeviceType::Metal>();
            #endif

                default:
                    throw std::invalid_argument( std::format(
                        "ModelHandle: this build of Mila runs no language model on {}", deviceTypeToString( device_type ) ) );
            }
        }

        template<DeviceType TDeviceType>
        LoadedModel loadOn( const InstalledModel& installed, const DeploymentRequest& request )
        {
            constexpr TensorDataType kPrecision = DeviceTypeTraits<TDeviceType>::language_model_precision;

            LoadedModel loaded;

            Architectures::visit( installed.architecture(), [&]( auto named )
                {
                    using Architecture = typename decltype( named )::type;

                    auto typed = Architecture::template Model<TDeviceType, kPrecision>::load(
                        installed.weightsPath(), request );

                    loaded.plan.emplace( typed->getDeploymentPlan() );
                    loaded.prefix_reuse = typed->promptPrefixReuse();
                    loaded.model = std::move( typed );
                } );

            return loaded;
        }

        void addOnce( std::vector<DeviceType>& types, DeviceType type )
        {
            if ( std::ranges::find( types, type ) == types.end() )
                types.push_back( type );
        }

        std::string namesOf( const std::vector<DeviceType>& types )
        {
            std::string names;

            for ( const DeviceType type : types )
            {
                if ( !names.empty() )
                    names += ", ";

                names += deviceTypeToString( type );
            }

            return names;
        }
    }

    std::optional<OpenRefusal> ModelHandle::check(
        std::string_view name, const DeploymentRequest& request, const ModelStore& store )
    {
        const std::optional<InstalledModel> installed = InstalledModel::find( name, store );

        if ( !installed )
        {
            return OpenRefusal{ OpenRefusal::Reason::NotInstalled, std::string( name ),
                std::format( "no model named '{}' is installed", name ) };
        }

        return check( *installed, request );
    }

    std::optional<OpenRefusal> ModelHandle::check( const InstalledModel& installed, const DeploymentRequest& request )
    {
        const std::string& name = installed.name();
        const auto& record = installed.stored().record;

        if ( !installed.isComplete() )
        {
            return OpenRefusal{ OpenRefusal::Reason::FilesMissing, name,
                std::format( "'{}' is installed but its files are missing, so it cannot be loaded", name ) };
        }

        if ( !Architectures::contains( installed.architecture() ) )
        {
            return OpenRefusal{ OpenRefusal::Reason::ArchitectureNotDeployed, name,
                std::format( "'{}' has architecture '{}', which this build of Mila cannot deploy; it deploys {}",
                    name, installed.architecture(), Architectures::names() ) };
        }

        if ( installed.capabilities().instruct )
        {
            try
            {
                (void)chatProtocolFor( installed.architecture(), record.architecture_version );
            }
            catch ( const std::invalid_argument& error )
            {
                // Family-blind: an architecture with one template takes an empty version as that one, so an
                // empty version refused is one whose record does not say which of several it is.
                if ( record.architecture_version.empty() )
                {
                    return OpenRefusal{ OpenRefusal::Reason::ChatTemplateNotRecorded, name, std::format(
                        "'{}' does not record which version of '{}' it is, which decides the chat template it was "
                        "trained with: {}", name, installed.architecture(), error.what() ) };
                }

                return OpenRefusal{ OpenRefusal::Reason::NoChatTemplate, name,
                    std::format( "'{}' was trained with a chat template this build does not have: {}", name, error.what() ) };
            }
        }

        const std::optional<WeightQuantization> stored = installed.storedWeightQuantization();

        if ( !stored )
        {
            return OpenRefusal{ OpenRefusal::Reason::WeightFormatUnknown, name, std::format(
                "'{}' stores its weights as '{}', which this build of Mila does not read", name, record.weight_quantization ) };
        }

        if ( request.isWeightQuantizationNamed() && request.getWeightQuantization() != *stored )
        {
            // The load's own rule, asked here so it refuses before anything is released.
            const std::string_view scheme = record.weight_quantization == "none"
                ? std::string_view{} : std::string_view( record.weight_quantization );

            try
            {
                requireStoredQuantizationMatches(
                    "ModelHandle::check", installed.weightsPath().string(), scheme, request.getWeightQuantization() );
            }
            catch ( const std::runtime_error& error )
            {
                return OpenRefusal{ OpenRefusal::Reason::WeightFormatNotDerivable, name, error.what() };
            }
        }

        if ( !request.isContextLengthAutomatic() )
        {
            const dim_t maximum = trainedMaximum( installed );

            if ( maximum > 0 && request.getContextLength() > maximum )
            {
                return OpenRefusal{ OpenRefusal::Reason::ContextPastTrainedMaximum, name, std::format(
                    "'{}' was trained for a context of at most {} tokens, and {} was asked for",
                    name, maximum, request.getContextLength() ) };
            }
        }

        return std::nullopt;
    }

    std::expected<DeviceType, DeploymentRefusal> ModelHandle::deviceTypeFor( const DeploymentRequest& request )
    {
        const std::vector<DeviceReading> present = DeviceReading::describePresent();
        const AllowedDevices& allowed = request.getAllowedDevices();

        std::vector<DeviceType> present_types;
        std::vector<DeviceType> allowed_types;

        for ( const DeviceReading& device : present )
        {
            if ( device.device.type == DeviceType::Cpu )
                continue;

            addOnce( present_types, device.device.type );

            if ( allowed.allows( device ) )
                addOnce( allowed_types, device.device.type );
        }

        // Nothing present allowed: the planner names what was asked for and is not here, against the devices that are.
        const std::vector<DeviceType>& types = allowed_types.empty() ? present_types : allowed_types;

        if ( types.empty() )
        {
            std::vector<std::string> absent = allowed.absentFrom( present );
            const DeploymentRefusal::Reason reason = absent.empty()
                ? DeploymentRefusal::Reason::NoDeviceToPlanOn
                : DeploymentRefusal::Reason::AllowedDeviceAbsent;

            return std::unexpected( DeploymentRefusal( reason, {}, {}, std::move( absent ) ) );
        }

        if ( types.size() > 1 )
        {
            throw std::invalid_argument( std::format(
                "ModelHandle: the devices allowed ({}) are of {} device types, {}; a model runs on one",
                allowed.toString(), types.size(), namesOf( types ) ) );
        }

        return types.front();
    }

    ModelHandle ModelHandle::open( std::string_view name, const DeploymentRequest& request, const ModelStore& store )
    {
        const InstalledModel installed = checkedNamed( name, request, store );

        // A base model has no template to render, so it has no protocol (ModelHandle.md 3.5).
        std::unique_ptr<Conversation::Protocol> protocol = installed.capabilities().instruct
            ? chatProtocolFor( installed.architecture(), installed.stored().record.architecture_version )
            : nullptr;

        std::shared_ptr<Data::BpeTokenizer> tokenizer = installed.loadTokenizer();

        const std::expected<DeviceType, DeploymentRefusal> device_type = deviceTypeFor( request );

        if ( !device_type )
        {
            throw DeploymentRefusedError( "ModelHandle::open", device_type.error() );
        }

        LoadedModel loaded = onDeviceType( *device_type, [&]<DeviceType TDeviceType>()
            {
                return loadOn<TDeviceType>( installed, request );
            } );

        return ModelHandle( installed, std::move( loaded.model ), std::move( tokenizer ), std::move( protocol ),
            std::move( *loaded.plan ), loaded.prefix_reuse );
    }

    std::expected<DeploymentPlans, DeploymentRefusal> ModelHandle::planFor(
        std::string_view name, const DeploymentRequest& request, const ModelStore& store )
    {
        const InstalledModel installed = checkedNamed( name, request, store );
        const std::expected<DeviceType, DeploymentRefusal> device_type = deviceTypeFor( request );

        if ( !device_type )
        {
            return std::unexpected( device_type.error() );
        }

        return onDeviceType( *device_type, [&]<DeviceType TDeviceType>()
            {
                return planDeployment<TDeviceType, DeviceTypeTraits<TDeviceType>::language_model_precision>(
                    installed, request );
            } );
    }

    std::optional<std::vector<WeightQuantization>> ModelHandle::fittingWeightQuantizations(
        const InstalledModel& model, const DeploymentRequest& request )
    {
        const std::expected<DeviceType, DeploymentRefusal> device_type = deviceTypeFor( request );

        if ( !device_type )
        {
            return std::vector<WeightQuantization>{};
        }

        return onDeviceType( *device_type, [&]<DeviceType TDeviceType>()
            {
                return Mila::Deployment::fittingWeightQuantizations<TDeviceType,
                    DeviceTypeTraits<TDeviceType>::language_model_precision>( model, request );
            } );
    }

    ModelHandle::ModelHandle( ModelHandle&& ) noexcept = default;

    ModelHandle& ModelHandle::operator=( ModelHandle&& ) noexcept = default;

    ModelHandle::~ModelHandle() = default;

    ILanguageModel& ModelHandle::model() noexcept
    {
        return *model_;
    }

    Data::BpeTokenizer& ModelHandle::tokenizer() const noexcept
    {
        return *tokenizer_;
    }

    const Conversation::Protocol* ModelHandle::protocol() const noexcept
    {
        return protocol_.get();
    }

    const ModelCapabilities& ModelHandle::capabilities() const noexcept
    {
        return capabilities_;
    }

    PromptPrefixReuse ModelHandle::promptPrefixReuse() const noexcept
    {
        return prefix_reuse_;
    }

    const DeploymentPlan& ModelHandle::plan() const noexcept
    {
        return plan_;
    }

    const InstalledModel& ModelHandle::installed() const noexcept
    {
        return installed_;
    }

    const std::string& ModelHandle::name() const noexcept
    {
        return installed_.name();
    }

    void ModelHandle::setMemoryHold( MemoryHold hold )
    {
        DeviceRegistry::instance().getDevice( model_->getDeviceId() )->setMemoryHold( hold );
    }

    ModelHandle::ModelHandle(
        InstalledModel installed,
        std::unique_ptr<ILanguageModel> model,
        std::shared_ptr<Data::BpeTokenizer> tokenizer,
        std::unique_ptr<Conversation::Protocol> protocol,
        DeploymentPlan plan,
        PromptPrefixReuse prefix_reuse )
        : installed_( std::move( installed ) ),
          capabilities_( installed_.capabilities() ),
          model_( std::move( model ) ),
          tokenizer_( std::move( tokenizer ) ),
          protocol_( std::move( protocol ) ),
          plan_( std::move( plan ) ),
          prefix_reuse_( prefix_reuse )
    {
    }

    dim_t ModelHandle::trainedMaximum( const InstalledModel& installed )
    {
        try
        {
            return static_cast<dim_t>( WeightsReader( installed.weightsPath() ).getWeightsMetadata().max_seq_length );
        }
        catch ( const std::exception& )
        {
            return 0;
        }
    }

    InstalledModel ModelHandle::checkedNamed( std::string_view name, const DeploymentRequest& request, const ModelStore& store )
    {
        std::optional<InstalledModel> installed = InstalledModel::find( name, store );

        if ( !installed )
        {
            throw OpenRefusedError( *check( name, request, store ) );
        }

        if ( auto refusal = check( *installed, request ) )
        {
            throw OpenRefusedError( std::move( *refusal ) );
        }

        return std::move( *installed );
    }
}
