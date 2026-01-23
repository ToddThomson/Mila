add_executable( Tokenize
	"Tokenize.cpp"
)

set_target_properties( Tokenize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Tools"
)

target_link_libraries( Tokenize PRIVATE Mila )