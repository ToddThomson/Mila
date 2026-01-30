add_executable( tokenize
	"Tokenize.cpp"
)

# Output directly to the Data/Tools directory where it's used
set_target_properties( tokenize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Data/Tools/$<CONFIG>"
)

target_link_libraries( tokenize PRIVATE Mila )

install( TARGETS tokenize
    RUNTIME DESTINATION bin
)