find_package(Doxygen REQUIRED)

if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
    set(DOXYGEN_PROJECT_NAME "Mila")
    set(DOXYGEN_PROJECT_BRIEF "Deep Neural Network Library")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_EXTRACT_PACKAGE YES)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_EXTRACT_LOCAL_METHODS YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_SOURCE_DIR}/README.md)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_SORT_BRIEF_DOCS YES)
    set(DOXYGEN_SORT_MEMBERS_CTORS_1ST YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_QT_AUTOBRIEF YES)
    set(DOXYGEN_WARN_NO_PARAMDOC YES)
    set(DOXYGEN_INCLUDE_GRAPH NO)
    set(DOXYGEN_INCLUDED_BY_GRAPH NO)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_HAVE_DOT YES)
    set(DOXYGEN_DOT_GRAPH_MAX_NODES 100)
    
    doxygen_add_docs(docs
        ${CMAKE_SOURCE_DIR}/README.md
        ${CMAKE_SOURCE_DIR}/Mila/Src
        COMMENT "Generating API documentation with Doxygen"
    )
    
    set_property(TARGET docs PROPERTY FOLDER "Docs")
    message(STATUS "Doxygen documentation target 'docs' created")
    message(STATUS "Documentation will be generated in: ${CMAKE_BINARY_DIR}/docs")
else()
    message(WARNING "Doxygen not found - Documentation will not be built")
endif()