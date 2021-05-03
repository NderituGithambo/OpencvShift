// Author Mark Mwai
// Date 28/4/2021

#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>

using namespace cv::xfeatures2d;
using namespace cv;
using std::cout;
using std::endl;
std::string image_path = "C:/Users/Francis pc/Pictures/Image Processing/TestImages/Test34face.png";
std::string image_path2 = "C:/Users/Francis pc/Pictures/Image Processing/TestImages/Test34Big.png";
cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

cv::Mat img2 = imread(image_path2, IMREAD_GRAYSCALE);

void using_loweRatio(std::vector<KeyPoint> kp1, std::vector<KeyPoint> kp2,
    Mat des1, Mat des2)
{
    //Match descriptor vectors with a FLANN based Matcher.   Using norm_l2 as SURF is a flloating point descriptor
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<DMatch> knn_matches;
    matcher->match(des1, des2, knn_matches);

    //Use Lowe's ratio test to filter matches
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    double max_dist = 0; double min_dist = 100;
    for (size_t i = 0; i < des1.rows; i++)
    {
        double dist = knn_matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
       
    }
    double min = knn_matches[0].distance;
    int min_i = 0;
    for (int i = 0; i < knn_matches.size(); i++)
    {
        for (int j = 0; j < knn_matches.size(); j++)
        {
            if (knn_matches[j].distance < min)
            {
                min = knn_matches[j].distance;
                min_i = j;
            }
            
        }
        good_matches.push_back(knn_matches[min_i]);
        knn_matches.erase(knn_matches.begin() + min_i);
        min = knn_matches[0].distance;
        min_i = 0;
    }




    //After finding matches draw the matches
    Mat img_matches;
    drawMatches(src, kp1, img2, kp2, good_matches, img_matches, Scalar::all(-1),
        Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //Display detected matches
    imshow("Good Matches", img_matches);
}
int main(int argc, char* argv[])
{
    const char* image_window = "Matched Keypoints";

    namedWindow(image_window, WINDOW_FREERATIO);

    

    if (src.empty() || img2.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
    //Create a feature Detector
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute(src, noArray(), keypoints, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

  //using_loweRatio(keypoints, keypoints, descriptors1, descriptors2);

    // matching descriptors
    Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L2, true));
    std::vector<DMatch> matches;
    Mat src_clone = imread(image_path2);
    matcher->match(descriptors1, descriptors2, matches);
   // drawKeypoints(src_clone, keypoints, src_clone, (0, 225, 0));
    
    Mat img_matches;
    drawMatches(src, keypoints, img2, keypoints2, matches, img_matches);
    imshow(image_window,img_matches);

    waitKey(0);
    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif