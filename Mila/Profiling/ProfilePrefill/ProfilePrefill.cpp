#include <iostream>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <cstdint>
//#include <cuda_profiler_api.h>

//import Mila;
//import Profiling.NvtxRange;

int main()
{
    //Mila::initialize();

    //using namespace Mila::Dnn;
    //using namespace Mila::Dnn::Compute;

    //using LlamaModelBF16Type = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>;

    std::filesystem::path model_path =
        std::filesystem::path( MODELS_DIR ) / "llama" / "llama32_3b_bf16.bin";

    constexpr int64_t context_length = 4096;
    constexpr int64_t seq_len = 128;

    if ( !std::filesystem::exists( model_path ) )
    {
        std::cerr << "Error: Model file not found: " << model_path << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Loading model from: " << model_path << "\n";

    //const DeviceId device{ DeviceType::Cuda, 0 };

    //auto model = LlamaModelBF16Type::fromPretrained(
    //    model_path,
    //    context_length,
    //    device,
    //    /*strict=*/true );

    //std::cout << model->toString();
    //std::cout << model->getMemoryStats().toString() << "\n";

    std::vector<int32_t> tokens( seq_len, 1 );

    std::cout << "Warming up...\n";
    //model->profilePrefill( tokens );

    std::cout << "Press Enter to begin profiling...\n";
    std::cin.get();

    std::cout << "Profiling prefill...\n";

    //cudaProfilerStart();

    {
        //Mila::Profiling::NvtxRange range( "Mila::Prefill" );
        //model->profilePrefill( tokens );
    }

    //cudaProfilerStop();

    /*{
        Mila::Profiling::NvtxRange range( "Mila::Prefill" );
        model->profilePrefill( tokens );
    }*/

    std::cout << "Done.\n";
    return EXIT_SUCCESS;
}