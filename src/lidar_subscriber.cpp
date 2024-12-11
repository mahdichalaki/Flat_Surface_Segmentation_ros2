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
#include <Eigen/Dense>

class FlatSurfaceSegmentation : public rclcpp::Node
{
public:
    FlatSurfaceSegmentation()
        : Node("flat_surface_segmentation")
    {
        // Initialize publishers and subscribers
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/scan/points", 10, std::bind(&FlatSurfaceSegmentation::pointCloudCallback, this, std::placeholders::_1));

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

        // Step 1: Preprocess the Point Cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = preprocessPointCloud(cloud);

        // Step 2: Iterative Plane Segmentation
        int cluster_index = 0;
        while (filtered_cloud->size() > 100) { // Stop when remaining cloud is too small
            pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

            if (!segmentLargestPlane(filtered_cloud, plane_cloud, coefficients)) {
                break; // No more planes found
            }

            RCLCPP_INFO(this->get_logger(), "Segmented plane %d with %zu points.", cluster_index, plane_cloud->size());

            // Step 3: Process and Publish the Plane
            processAndPublishPlane(plane_cloud, coefficients, msg->header.frame_id, cluster_index);
            cluster_index++;
        }

        RCLCPP_INFO(this->get_logger(), "Finished processing %d planes.", cluster_index);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        // Remove ground points
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.6, 20.0);
        pass.filter(*non_ground_cloud);

        // Downsample the Point Cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(non_ground_cloud);
        voxel_grid.setLeafSize(0.02f, 0.02f, 0.02f);
        voxel_grid.filter(*downsampled_cloud);

        // Remove Noise
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(downsampled_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_cloud);

        return filtered_cloud;
    }

    bool segmentLargestPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud,
                             pcl::ModelCoefficients::Ptr &coefficients)
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No plane found in the point cloud.");
            return false;
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane_cloud);

        // Remove the plane points from the original cloud
        extract.setNegative(true);
        extract.filter(*cloud);

        return true;
    }

    void processAndPublishPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud,
                                const pcl::ModelCoefficients::Ptr &coefficients,
                                const std::string &frame_id,
                                int cluster_index)
    {
        // Project points onto the plane
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(plane_cloud);
        proj.setModelCoefficients(coefficients);
        proj.filter(*cloud_projected);

        // Calculate Concave Hull
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ConcaveHull<pcl::PointXYZ> chull;
        chull.setInputCloud(cloud_projected);
        chull.setAlpha(0.1);
        chull.reconstruct(*cloud_hull);

        RCLCPP_INFO(this->get_logger(), "Processed plane %d with boundary of %zu points.", cluster_index, cloud_hull->size());

        // Publish Markers
        publishBoundaryMarkers(cloud_hull, frame_id, cluster_index);
    }

    void publishBoundaryMarkers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                const std::string &frame_id,
                                int cluster_index)
    {
        visualization_msgs::msg::Marker line_strip;
        line_strip.header.frame_id = frame_id;
        line_strip.header.stamp = this->now();
        line_strip.ns = "boundary_cluster";
        line_strip.id = cluster_index;
        line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_strip.action = visualization_msgs::msg::Marker::ADD;

        line_strip.scale.x = 0.01;
        line_strip.color.r = 1.0f;
        line_strip.color.g = 0.0f;
        line_strip.color.b = 0.0f;
        line_strip.color.a = 1.0f;

        for (const auto &point : cloud_hull->points) {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            line_strip.points.push_back(p);
        }

        // Close the line strip
        if (!cloud_hull->points.empty()) {
            geometry_msgs::msg::Point first_point;
            first_point.x = cloud_hull->points.front().x;
            first_point.y = cloud_hull->points.front().y;
            first_point.z = cloud_hull->points.front().z;
            line_strip.points.push_back(first_point);
        }

        marker_pub_->publish(line_strip);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FlatSurfaceSegmentation>());
    rclcpp::shutdown();
    return 0;
}
