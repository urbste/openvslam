# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/SE3hom.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_hom.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_container.h
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_container_hom.h
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_pose_opt_edge.h
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_pose_opt_edge_hom.h
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_reproj_edge.h
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_nullspace_reproj_edge.h
               ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_pose_opt_edge.h
               ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_reproj_edge.h
               ${CMAKE_CURRENT_SOURCE_DIR}/SE3hom.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_hom.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_container.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_container_hom.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_pose_opt_edge.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_pose_opt_edge_hom.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_reproj_edge.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/perspective_nullspace_reproj_edge.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_pose_opt_edge.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_reproj_edge.cc)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/optimize/g2o/se3)
