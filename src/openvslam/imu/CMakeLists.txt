# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/imu_utils.h
               ${CMAKE_CURRENT_SOURCE_DIR}/config.cc
	       ${CMAKE_CURRENT_SOURCE_DIR}/imu_utils.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/data.cc)
