cmake_minimum_required(VERSION 3.5)
project(lidar_path_planner_cpp)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

# Add the PCL include directories and link libraries
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add the executable
add_executable(lidar_subscriber src/lidar_subscriber.cpp)
ament_target_dependencies(lidar_subscriber rclcpp sensor_msgs pcl_conversions)

# Link against PCL
target_link_libraries(lidar_subscriber ${PCL_LIBRARIES})

# Install the executable
install(TARGETS lidar_subscriber
  DESTINATION lib/${PROJECT_NAME})

# Include directories
ament_export_include_directories(include)

# Export dependencies
ament_export_dependencies(rclcpp sensor_msgs pcl_conversions)

# Ament package macros
ament_package()


