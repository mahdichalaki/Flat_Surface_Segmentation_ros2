#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/concave_hull.h>
#include <Eigen/Dense>
#include <pcl/filters/passthrough.h>
#include <geometry_msgs/msg/pose_array.hpp>

// FlatSurfaceSegmentation: A ROS2 node for detecting and processing flat surfaces
class FlatSurfaceSegmentation : public rclcpp::Node {
public:
    // Constructor: Initialize the node and its components
    FlatSurfaceSegmentation()
        : Node("flat_surface_segmentation"), process_request_(false), current_plane_index_(-1) {
        // Subscribe to a point cloud topic
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/scan/points", 10,
            std::bind(&FlatSurfaceSegmentation::pointCloudCallback, this, std::placeholders::_1));
        
        // Service to request processing the incoming point cloud
        request_processing_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/request_processing",
            std::bind(&FlatSurfaceSegmentation::handleRequestProcessing, this, std::placeholders::_1, std::placeholders::_2));

        // Service to switch to the next detected plane
        next_plane_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/next_plane",
            std::bind(&FlatSurfaceSegmentation::handleNextPlaneRequest, this, std::placeholders::_1, std::placeholders::_2));

        // Publishers for visualization and processing results
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/segmentation_marker", 10);
        hull_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/flat_surface_hull", 10);
        waypoints_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/flat_surface_waypoints", 10);
        
        // Timer to periodically publish results
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&FlatSurfaceSegmentation::publishStoredResults, this));

        RCLCPP_INFO(this->get_logger(), "FlatSurfaceSegmentation node initialized.");
    }

private:
    // Data structure to store information about each detected plane
    struct PlaneData {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;       // Points belonging to the plane
        pcl::ModelCoefficients::Ptr coefficients;       // Plane equation coefficients
        pcl::PointCloud<pcl::PointXYZ>::Ptr hull;       // Concave hull of the plane
    };

    std::vector<PlaneData> planes_;                     // List of detected planes
    int current_plane_index_;                           // Index of the current plane
    bool process_request_;                              // Flag to indicate processing request

    // ROS2 components: subscribers, services, publishers, and timers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr request_processing_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr next_plane_service_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr hull_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr waypoints_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Callback to handle incoming point clouds
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!process_request_) {
            return; // Skip if no request is active
        }

        process_request_ = false; // Reset processing flag

        // Convert ROS PointCloud2 message to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud.");
            return;
        }

        // Preprocess the point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = preprocessPointCloud(cloud);
        planes_.clear(); // Clear previous plane data

        // Detect and segment planes from the point cloud
        while (filtered_cloud->size() > 100) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

            if (!segmentLargestPlane(filtered_cloud, plane_cloud, coefficients)) {
                break; // Exit if no more planes can be detected
            }

            // Generate a concave hull for the detected plane
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::ConcaveHull<pcl::PointXYZ> chull;
            chull.setInputCloud(plane_cloud);
            chull.setAlpha(0.1); // Set alpha parameter for concave hull
            chull.reconstruct(*cloud_hull);

            // Store plane data
            planes_.push_back({plane_cloud, coefficients, cloud_hull});
        }

        // Log results or warn if no planes detected
        if (planes_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No planes detected.");
        } else {
            current_plane_index_ = 0;
            RCLCPP_INFO(this->get_logger(), "Detected %zu planes.", planes_.size());
        }
    }

    // Service handler for processing requests
    void handleRequestProcessing(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        process_request_ = true; // Enable processing
        response->success = true;
        response->message = "Processing request received.";
    }

    // Service handler for switching to the next plane
    void handleNextPlaneRequest(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        if (planes_.empty()) {
            response->success = false;
            response->message = "No planes available.";
            return;
        }

        if (current_plane_index_ < 0 || current_plane_index_ >= static_cast<int>(planes_.size() - 1)) {
            response->success = false;
            response->message = "All planes processed.";
            return;
        }

        current_plane_index_++; // Move to the next plane
        response->success = true;
        response->message = "Moved to the next plane.";
    }

    // Helper function to preprocess the point cloud (filtering and downsampling)
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        // Apply a pass-through filter to remove distant points
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.6, 20.0); // Keep points within this range
        pass.filter(*filtered_cloud);

        // Downsample using a voxel grid filter
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(filtered_cloud);
        voxel_grid.setLeafSize(0.02f, 0.02f, 0.02f); // Voxel size
        voxel_grid.filter(*filtered_cloud);

        // Remove outliers using statistical filtering
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(filtered_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_cloud);

        return filtered_cloud;
    }

    // More functions like `segmentLargestPlane`, `publishStoredResults` with similar comments...
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FlatSurfaceSegmentation>());
    rclcpp::shutdown();
    return 0;
}
