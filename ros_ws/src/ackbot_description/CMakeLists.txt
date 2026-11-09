cmake_minimum_required(VERSION 3.8)
project(ackbot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install directories
install(
  DIRECTORY urdf launch rviz
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*"
)

ament_package()
