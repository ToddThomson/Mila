class MyCustomLayerPlugin : public IModulePlugin
{
public:
    PluginInfo getInfo() const override
    {
        return {
            .name = "MyCustomLayer",
            .version = "1.0.0",
            .mila_api_version = "0.1.0",
            .supported_devices = {"Cpu", "Cuda"},
            .supported_precisions = {"float32", "float64"}
        };
    }

    bool canHandle( const std::string& module_type ) const override
    {
        return module_type == "MyCustomLayer";
    }

    std::unique_ptr<IModule> createFromArchive(
        const std::string& device_type,
        const std::string& precision,
        ModelArchive& archive,
        const std::string& module_name,
        void* exec_context ) const override
    {
        // Dispatch based on device and precision
        if (device_type == "Cuda" && precision == "float32")
        {
            auto ctx = static_cast<ExecutionContext<DeviceType::Cuda>*>(exec_context);
            return MyCustomLayer<DeviceType::Cuda, TensorDataType::Float32>
                ::fromArchive_( archive, module_name,
                    std::shared_ptr<ExecutionContext<DeviceType::Cuda>>( ctx ) );
        }
        // ... other combinations

        throw std::runtime_error( "Unsupported device/precision combination" );
    }

    void registerOperations( OperationRegistry& registry ) override
    {
        // Register backend compute kernels
        registry.registerOperation<DeviceType::Cuda>(
            "MyCustomLayerForward",
            std::make_unique<MyCustomLayerForwardOp>()
        );
        registry.registerOperation<DeviceType::Cuda>(
            "MyCustomLayerBackward",
            std::make_unique<MyCustomLayerBackwardOp>()
        );
    }
};

// Plugin entry point (C linkage to avoid name mangling)
extern "C" {
    MILA_PLUGIN_EXPORT IModulePlugin* mila_create_plugin()
    {
        return new MyCustomLayerPlugin();
    }
}