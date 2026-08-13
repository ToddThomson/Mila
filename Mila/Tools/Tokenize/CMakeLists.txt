add_executable( tokenize
	"Src/main.cpp" )

target_sources( tokenize
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Commands.ixx"
        "Src/Args.ixx")

# Output directly to the Data/Tools directory where it's used
set_target_properties( tokenize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Data/Tools/$<CONFIG>"
)

target_link_libraries( tokenize PRIVATE Mila )