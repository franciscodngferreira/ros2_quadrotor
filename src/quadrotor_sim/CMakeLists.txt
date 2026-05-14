cmake_minimum_required(VERSION 3.8)
project(quadrotor_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz_bridge REQUIRED)

# Install ROS2 resources
install(DIRECTORY
  models
  worlds
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

ament_package()