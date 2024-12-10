#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/concave_hull.h>

class FlatSurfaceSegmentation : public rclcpp::Node
{
public:
    FlatSurfaceSegmentation()
        : Node("flat_surface_segmentation")
    {
        // Initialize publishers and subscribers
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/scan/points", 10, std::bind(&FlatSurfaceSegmentation::pointCloudCallback, this, std::placeholders::_1));
        
        segmented_surface_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/flat_surface", 10);
        hull_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/flat_surface_hull", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/segmentation_marker", 10);

        RCLCPP_INFO(this->get_logger(), "FlatSurfaceSegmentation node initialized.");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Point cloud received with %u points.", msg->width * msg->height);

        // Convert ROS2 PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);
        RCLCPP_INFO(this->get_logger(), "Converted PointCloud2 to PCL format with %zu points.", cloud->size());

        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud.");
            return;
        }

        // Step 1: Remove Ground Points using PassThrough Filter
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z"); // Filter based on height (z-axis)
        pass.setFilterLimits(-0.6, 20.0); // Set limits to remove ground
        pass.filter(*non_ground_cloud);
        RCLCPP_INFO(this->get_logger(), "Filtered ground points. Remaining points: %zu.", non_ground_cloud->size());

        if (non_ground_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "All points were filtered out by PassThrough.");
            return;
        }

        // Step 2: Downsample the Point Cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(non_ground_cloud);
        voxel_grid.setLeafSize(0.02f, 0.02f, 0.02f); // Adjust leaf size as needed
        voxel_grid.filter(*downsampled_cloud);
        RCLCPP_INFO(this->get_logger(), "Downsampled point cloud to %zu points.", downsampled_cloud->size());

        // Step 3: Remove Noise
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(downsampled_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_cloud);
        RCLCPP_INFO(this->get_logger(), "Filtered point cloud to %zu points after noise removal.", filtered_cloud->size());

        // Step 4: Segment the Largest Plane
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        seg.setInputCloud(filtered_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No plane found in the point cloud.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Segmented plane with %zu inliers.", inliers->indices.size());

        // Step 5: Extract the Plane
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filtered_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false); // Keep inliers
        extract.filter(*plane_cloud);
        RCLCPP_INFO(this->get_logger(), "Extracted flat surface with %zu points.", plane_cloud->size());

        // Step 6: Project Plane Points
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(plane_cloud);
        proj.setModelCoefficients(coefficients);
        proj.filter(*cloud_projected);
        RCLCPP_INFO(this->get_logger(), "Projected inliers onto the plane with %zu points.", cloud_projected->size());

        // Step 7: Calculate and Publish Concave Hull
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ConcaveHull<pcl::PointXYZ> chull;
        chull.setInputCloud(cloud_projected);
        chull.setAlpha(0.1); // Adjust alpha for desired concaveness
        chull.reconstruct(*cloud_hull);
        RCLCPP_INFO(this->get_logger(), "Calculated concave hull with %zu points.", cloud_hull->size());

        // Publish the Concave Hull
        sensor_msgs::msg::PointCloud2 hull_msg;
        pcl::toROSMsg(*cloud_hull, hull_msg);
        hull_msg.header.frame_id = msg->header.frame_id; // Use the same frame as input
        hull_msg.header.stamp = msg->header.stamp;
        hull_pub_->publish(hull_msg);
        RCLCPP_INFO(this->get_logger(), "Published concave hull to /flat_surface_hull.");

        // Step 8: Publish the Segmented Plane
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*plane_cloud, output_msg);
        output_msg.header.frame_id = msg->header.frame_id;
        output_msg.header.stamp = msg->header.stamp;
        segmented_surface_pub_->publish(output_msg);
        RCLCPP_INFO(this->get_logger(), "Published segmented flat surface to /flat_surface.");

        // Optional: Publish a Marker for Visualization
        publishMarker(coefficients);
    }

    void publishMarker(const pcl::ModelCoefficients::Ptr &coefficients)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link"; // Adjust if necessary
        marker.header.stamp = this->now();
        marker.ns = "plane";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Define the plane's approximate position and orientation
        marker.pose.position.x = coefficients->values[0];
        marker.pose.position.y = coefficients->values[1];
        marker.pose.position.z = coefficients->values[2];
        marker.pose.orientation.w = 1.0;

        // Set the size and color of the marker
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 0.01; // Thickness of the plane
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.5;

        marker_pub_->publish(marker);
        RCLCPP_INFO(this->get_logger(), "Published marker for segmented plane.");
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_surface_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr hull_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FlatSurfaceSegmentation>());
    rclcpp::shutdown();
    return 0;
}
