cmake_minimum_required(VERSION 3.5)
project(lidar_path_planner_cpp)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(PCL REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)

# Add the PCL include directories and link libraries
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add the executable
add_executable(lidar_subscriber src/lidar_subscriber.cpp)
ament_target_dependencies(lidar_subscriber rclcpp sensor_msgs pcl_conversions geometry_msgs std_msgs std_srvs visualization_msgs tf2_ros tf2_eigen tf2_geometry_msgs pcl_ros)

# Link against PCL and Eigen3
target_link_libraries(lidar_subscriber ${PCL_LIBRARIES} Eigen3::Eigen)

# Install the executable
install(TARGETS lidar_subscriber
  DESTINATION lib/${PROJECT_NAME})

# Include directories
ament_export_include_directories(include)

# Export dependencies
ament_export_dependencies(rclcpp sensor_msgs pcl_conversions)

# Ament package macros
ament_package()


