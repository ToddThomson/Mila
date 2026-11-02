# Measurement harness: header-only helpers that build a network from a weights file and score text
# through it. Shared by the tests and Tools/ContextProfile; ships with neither. Include after
# `import Mila;` as "Measurement/<header>.h".
add_library( MilaMeasurement INTERFACE )

target_include_directories( MilaMeasurement INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/.." )
