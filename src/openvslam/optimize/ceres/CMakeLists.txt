# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/autodiff.h
	       ${CMAKE_CURRENT_SOURCE_DIR}/eigen.h
               ${CMAKE_CURRENT_SOURCE_DIR}/fixed_array.h
               ${CMAKE_CURRENT_SOURCE_DIR}/fpclassify.h
	       ${CMAKE_CURRENT_SOURCE_DIR}/jet.h
               ${CMAKE_CURRENT_SOURCE_DIR}/macros.h
	       ${CMAKE_CURRENT_SOURCE_DIR}/manual_constructor.h
               ${CMAKE_CURRENT_SOURCE_DIR}/variadic_evaluate.h)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/optimize/ceres)

