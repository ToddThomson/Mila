add_executable( tokenize
	"Src/main.cpp" )

target_sources( tokenize
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Commands.ixx"
        "Src/Args.ixx")

# Output directly to the Data/Tools directory where it's used. MILA_DATA_DIR is
# ${PROJECT_SOURCE_DIR}/Data, so this stays inside Mila's own tree when Mila is a
# subproject; CMAKE_SOURCE_DIR would be the consumer's root.
set_target_properties( tokenize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${MILA_DATA_DIR}/Tools/$<CONFIG>"
)

target_link_libraries( tokenize PRIVATE Mila )