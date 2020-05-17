# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex.h
	       ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex4.h
               ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_container.h
               ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex4_container.h
	       ${CMAKE_CURRENT_SOURCE_DIR}/nullspace.h
               ${CMAKE_CURRENT_SOURCE_DIR}/nullspace_updater.h
               ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex4.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_container.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex4_container.cc)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/optimize/g2o)

# Append subdirectory
add_subdirectory(se3)
add_subdirectory(sim3)
