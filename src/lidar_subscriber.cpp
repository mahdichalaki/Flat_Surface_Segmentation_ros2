#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/io/pcd_io.h>
#include <unistd.h>

class LidarSubscriber : public rclcpp::Node {
public:
    LidarSubscriber() : Node("lidar_subscriber") {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            RCLCPP_INFO(this->get_logger(), "Current working directory: %s", cwd);
        }
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/scan/points", 10,
            std::bind(&LidarSubscriber::callback, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Subscribed to /scan/points");
    }

private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);
        RCLCPP_INFO(this->get_logger(), "Initial cloud size: %zu", cloud->points.size());

        // Remove noise with more lenient parameters
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cleaned(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(30);  // Reduced from 50
        sor.setStddevMulThresh(2.0);  // Increased from 1.0
        sor.filter(*cloud_cleaned);
        RCLCPP_INFO(this->get_logger(), "After noise removal: %zu points", cloud_cleaned->points.size());

        // Filter out ground points with adjusted limits
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_cleaned);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.0, 10.0);  // Changed from (0.1, 10.0) to include more points
        pass.filter(*non_ground_cloud);
        RCLCPP_INFO(this->get_logger(), "After ground filtering: %zu points", non_ground_cloud->points.size());

        // Segment the largest plane with adjusted threshold
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.05);  // Increased from 0.01
        seg.setInputCloud(non_ground_cloud);
        seg.segment(*inliers, *coefficients);

        // Extract objects with more lenient parameters
        pcl::PointCloud<pcl::PointXYZ>::Ptr objects_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(non_ground_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*objects_cloud);
        RCLCPP_INFO(this->get_logger(), "After plane segmentation: %zu points", objects_cloud->points.size());

        // Cluster with adjusted parameters
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(objects_cloud);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.1);     // Increased from 0.05
        ec.setMinClusterSize(50);       // Decreased from 100
        ec.setMaxClusterSize(50000);    // Increased from 25000
        ec.setSearchMethod(tree);
        ec.setInputCloud(objects_cloud);
        ec.extract(cluster_indices);

        int cluster_index = 0;
        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& index : indices.indices) {
                cluster->points.push_back(objects_cloud->points[index]);
            }
            cluster->width = cluster->points.size();
            cluster->height = 1;
            cluster->is_dense = true;

            // Create a unique filename for each cluster
            std::string filename = "cluster_" + std::to_string(cluster_index) + ".pcd";
            
            // Add error handling for file saving
            if (pcl::io::savePCDFileASCII(filename, *cluster) == -1) {
                RCLCPP_ERROR(this->get_logger(), "Failed to save cluster to file: %s", filename.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "Saved cluster to file: %s", filename.c_str());
            }
            
            cluster_index++;
        }

        RCLCPP_INFO(this->get_logger(), "Processed and saved %d clusters.", cluster_index);
    }


    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarSubscriber>());
    rclcpp::shutdown();
    return 0;
}
