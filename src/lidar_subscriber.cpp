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
#include <pcl/filters/passthrough.h>

class FlatSurfaceSegmentation : public rclcpp::Node {
public:
    FlatSurfaceSegmentation()
        : Node("flat_surface_segmentation"), process_request_(false), current_plane_index_(-1) {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/scan/points", 10,
            std::bind(&FlatSurfaceSegmentation::pointCloudCallback, this, std::placeholders::_1));
        
        request_processing_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/request_processing",
            std::bind(&FlatSurfaceSegmentation::handleRequestProcessing, this, std::placeholders::_1, std::placeholders::_2));

        next_plane_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/next_plane",
            std::bind(&FlatSurfaceSegmentation::handleNextPlaneRequest, this, std::placeholders::_1, std::placeholders::_2));

        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/segmentation_marker", 10);
        hull_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/flat_surface_hull", 10);

        // Timer for periodic publishing of results
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&FlatSurfaceSegmentation::publishStoredResults, this));

        RCLCPP_INFO(this->get_logger(), "FlatSurfaceSegmentation node initialized.");
    }

private:
    struct PlaneData {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::ModelCoefficients::Ptr coefficients;
        pcl::PointCloud<pcl::PointXYZ>::Ptr hull;
    };

    std::vector<PlaneData> planes_;
    int current_plane_index_;
    bool process_request_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr request_processing_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr next_plane_service_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr hull_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!process_request_) {
            return; // Skip processing if no request
        }

        process_request_ = false; // Reset flag
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = preprocessPointCloud(cloud);
        planes_.clear();

        while (filtered_cloud->size() > 100) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

            if (!segmentLargestPlane(filtered_cloud, plane_cloud, coefficients)) {
                break;
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::ConcaveHull<pcl::PointXYZ> chull;
            chull.setInputCloud(plane_cloud);
            chull.setAlpha(0.1);
            chull.reconstruct(*cloud_hull);

            planes_.push_back({plane_cloud, coefficients, cloud_hull});
        }

        if (planes_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No planes detected.");
        } else {
            current_plane_index_ = 0;
            RCLCPP_INFO(this->get_logger(), "Detected %zu planes.", planes_.size());
        }
    }

    void handleRequestProcessing(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        process_request_ = true;
        response->success = true;
        response->message = "Processing request received.";
    }

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

        current_plane_index_++;
        response->success = true;
        response->message = "Moved to the next plane.";
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.6, 20.0);
        pass.filter(*filtered_cloud);

        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(filtered_cloud);
        voxel_grid.setLeafSize(0.02f, 0.02f, 0.02f);
        voxel_grid.filter(*filtered_cloud);

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(filtered_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_cloud);

        return filtered_cloud;
    }

    bool segmentLargestPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud,
                             pcl::ModelCoefficients::Ptr &coefficients) {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            return false;
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane_cloud);

        extract.setNegative(true);
        extract.filter(*cloud);

        return true;
    }

    void publishStoredResults() {
        if (planes_.empty()) {
            return;
        }

        const auto &current_plane = planes_[current_plane_index_];

        // Publish current plane hull
        sensor_msgs::msg::PointCloud2 hull_msg;
        pcl::toROSMsg(*current_plane.hull, hull_msg);
        hull_msg.header.frame_id = current_plane.cloud->header.frame_id; // Use appropriate frame
        hull_msg.header.stamp = this->now();
        hull_pub_->publish(hull_msg);

        // Publish marker for visualization
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = current_plane.cloud->header.frame_id; // Use appropriate frame
        marker.header.stamp = this->now();
        marker.ns = "planes";
        marker.id = current_plane_index_;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.scale.x = 0.01;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        for (const auto &point : current_plane.hull->points) {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            marker.points.push_back(p);
        }

        marker_pub_->publish(marker);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FlatSurfaceSegmentation>());
    rclcpp::shutdown();
    return 0;
}
