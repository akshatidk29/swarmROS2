cmake_minimum_required(VERSION 3.8)
project(swarm_description)

find_package(ament_cmake REQUIRED)

# Install URDF, meshes, config, launch, worlds directories
install(DIRECTORY
  urdf
  launch
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
