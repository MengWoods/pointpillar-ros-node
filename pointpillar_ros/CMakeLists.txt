cmake_minimum_required(VERSION 3.1)
project(pointpillar_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  jsk_recognition_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install directories for launch files and models
install(DIRECTORY
  launch
  model
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/pointpillar_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

