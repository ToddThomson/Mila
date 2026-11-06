# Through the library, so that a rate is what an application gets from LlamaModel::load and generate.
add_executable( mila-metal-model-timing
    "ModelTiming.cpp" )

target_link_libraries( mila-metal-model-timing PRIVATE Mila )
