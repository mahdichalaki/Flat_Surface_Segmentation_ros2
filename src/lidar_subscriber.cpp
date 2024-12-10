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

        // Step 8: Process Boundary Points
        processHullPoints(cloud_hull, coefficients, msg);
    }

    void processHullPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                           const pcl::ModelCoefficients::Ptr &coefficients,
                           const sensor_msgs::msg::PointCloud2::SharedPtr &msg)
    {
        // Sort boundary points
        auto sorted_points = sortBoundaryPoints(cloud_hull);
        RCLCPP_INFO(this->get_logger(), "Sorted %zu boundary points.", sorted_points.size());

        // Simplify boundary (optional)
        auto simplified_points = simplifyBoundary(sorted_points, 5); // Adjust step for simplification
        RCLCPP_INFO(this->get_logger(), "Simplified boundary to %zu points.", simplified_points.size());

        // Convert to ROS waypoints
        std::vector<geometry_msgs::msg::Pose> waypoints;
        for (const auto &point : simplified_points) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = point.x;
            pose.position.y = point.y;
            pose.position.z = point.z;

            // Use plane coefficients for orientation
            Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
            Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), normal.normalized());
            pose.orientation.w = q.w();
            pose.orientation.x = q.x();
            pose.orientation.y = q.y();
            pose.orientation.z = q.z();

            waypoints.push_back(pose);
        }

        // Publish waypoints as a visualization marker
        publishWaypoints(waypoints, msg->header.frame_id);
    }

    std::vector<pcl::PointXYZ> sortBoundaryPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull)
    {
        Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
        for (const auto &point : cloud_hull->points) {
            centroid += Eigen::Vector3f(point.x, point.y, point.z);
        }
        centroid /= cloud_hull->points.size();

        std::vector<std::pair<float, pcl::PointXYZ>> polar_points;
        for (const auto &point : cloud_hull->points) {
            float angle = atan2(point.y - centroid.y(), point.x - centroid.x());
            polar_points.emplace_back(angle, point);
        }

        std::sort(polar_points.begin(), polar_points.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        std::vector<pcl::PointXYZ> sorted_points;
        for (const auto &pair : polar_points) {
            sorted_points.push_back(pair.second);
        }

        return sorted_points;
    }

    std::vector<pcl::PointXYZ> simplifyBoundary(const std::vector<pcl::PointXYZ> &points, int step)
    {
        std::vector<pcl::PointXYZ> simplified_points;
        for (size_t i = 0; i < points.size(); i += step) {
            simplified_points.push_back(points[i]);
        }
        return simplified_points;
    }

    void publishWaypoints(const std::vector<geometry_msgs::msg::Pose> &waypoints, const std::string &frame_id)
    {
        visualization_msgs::msg::Marker line_strip;
        line_strip.header.frame_id = frame_id; // Use the input point cloud's frame
        line_strip.header.stamp = this->now();
        line_strip.ns = "boundary";
        line_strip.id = 0;
        line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_strip.action = visualization_msgs::msg::Marker::ADD;

        line_strip.scale.x = 0.01; // Line width
        line_strip.color.r = 1.0f;
        line_strip.color.g = 0.0f;
        line_strip.color.b = 0.0f;
        line_strip.color.a = 1.0f;

        for (const auto &pose : waypoints) {
            geometry_msgs::msg::Point p;
            p.x = pose.position.x;
            p.y = pose.position.y;
            p.z = pose.position.z;
            line_strip.points.push_back(p);
        }

        // Ensure the line strip is closed by adding the first point at the end
        if (!waypoints.empty()) {
            geometry_msgs::msg::Point first_point;
            first_point.x = waypoints.front().position.x;
            first_point.y = waypoints.front().position.y;
            first_point.z = waypoints.front().position.z;
            line_strip.points.push_back(first_point);
        }

        marker_pub_->publish(line_strip);
        RCLCPP_INFO(this->get_logger(), "Published closed boundary waypoints as a line strip.");
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
