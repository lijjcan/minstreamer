#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace cv::dnn;

struct PosePoint {
    Point2f pos;
    float confidence;
};

// Pairs of keypoints for skeleton drawing based on COCO model
static const std::vector<std::pair<int, int>> POSE_PAIRS = {
    {1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
    {1,0}, {0,14}, {14,16}, {0,15}, {15,17}
};

std::vector<PosePoint> detectPose(Net &net, const Mat &frame) {
    const int inWidth = 368;
    const int inHeight = 368;
    const float thresh = 0.1f;

    Mat blob = blobFromImage(frame, 1.0/255, Size(inWidth, inHeight), Scalar(0,0,0), false, false);
    net.setInput(blob);
    Mat output = net.forward();

    int H = output.size[2];
    int W = output.size[3];
    int nPoints = output.size[1];

    std::vector<PosePoint> points(nPoints);
    for (int n=0; n<nPoints; ++n) {
        Mat probMap(H, W, CV_32F, output.ptr(0,n));
        Point2f p(-1,-1);
        double conf;
        minMaxLoc(probMap, 0, &conf, 0, &p);
        if (conf > thresh) {
            p.x *= frame.cols / static_cast<float>(W);
            p.y *= frame.rows / static_cast<float>(H);
        } else {
            p = Point2f(-1,-1);
        }
        points[n] = {p, static_cast<float>(conf)};
    }
    return points;
}

float angleBetween(const Point2f &a, const Point2f &b, const Point2f &c) {
    Point2f ab = a - b;
    Point2f cb = c - b;
    float dot = ab.x * cb.x + ab.y * cb.y;
    float normAb = std::sqrt(ab.x*ab.x + ab.y*ab.y);
    float normCb = std::sqrt(cb.x*cb.x + cb.y*cb.y);
    if (normAb * normCb == 0) return 0.f;
    float cosTheta = dot / (normAb * normCb);
    return std::acos(std::max(-1.f, std::min(1.f, cosTheta))) * 180 / CV_PI;
}

void drawPose(Mat &img, const std::vector<PosePoint> &points) {
    for (const auto &pair : POSE_PAIRS) {
        PosePoint p1 = points[pair.first];
        PosePoint p2 = points[pair.second];
        if (p1.pos.x>0 && p2.pos.x>0) {
            line(img, p1.pos, p2.pos, Scalar(0,255,0), 2);
            circle(img, p1.pos, 3, Scalar(0,0,255), -1);
            circle(img, p2.pos, 3, Scalar(0,0,255), -1);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <proto> <model>\n";
        return 1;
    }

    std::string proto = argv[1];
    std::string model = argv[2];
    Net net = readNetFromCaffe(proto, model);
    if (net.empty()) {
        std::cerr << "Failed to load network\n";
        return 1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return 1;
    }

    Mat frame;
    while (cap.read(frame)) {
        auto points = detectPose(net, frame);

        // Example evaluation: check elbow angle
        if (points[2].pos.x>0 && points[3].pos.x>0 && points[4].pos.x>0) {
            float angle = angleBetween(points[2].pos, points[3].pos, points[4].pos);
            putText(frame, cv::format("Elbow: %.1f", angle), Point(10,30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,0,0), 2);
            if (angle < 40 || angle > 160) {
                putText(frame, "Bad form", Point(10,60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,0,255), 2);
            }
        }

        drawPose(frame, points);
        imshow("Pose", frame);
        if (waitKey(1) == 27) break; // exit on ESC
    }
    return 0;
}

