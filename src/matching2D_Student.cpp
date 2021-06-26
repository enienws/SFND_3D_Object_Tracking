
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
            descSource.convertTo(descSource, CV_32F);
        
        if (descRef.type() != CV_32F)
            descRef.convertTo(descRef, CV_32F);
        
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // k nearest neighbors (k=2)

        int k = 2;
        double distRatio = 0.8;

        vector<vector<cv::DMatch>> knn_matches;
        //t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        // distance ratio filtering
        for (int i = 0; i < kPtsSource.size(); ++i) 
        {
            if (knn_matches[i][0].distance < distRatio * knn_matches[i][1].distance) 
            {
                matches.push_back(knn_matches[i][0]);
            }
        }
        //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << "SEL_KNN with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
// select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        //std::cout << "using BRISK descriptor" << std::endl;

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        //std::cout << "using BRIEF descriptor" << std::endl;
        //BRIEF descriptor implementation
        int bytes = 32;
        bool use_orientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create	(bytes, use_orientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
       // std::cout << "using ORB descriptor" << std::endl;
        //ORB descriptor implementation
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8; 
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType  scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        //std::cout << "using FREAK descriptor" << std::endl;

        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0f;
        int	nOctaves = 4;
        const std::vector<int> & selectedPairs = std::vector< int >();

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale,
                                                nOctaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        //std::cout << "using AKAZE descriptor" << std::endl;
//
        //AKAZE descriptor implementation
        cv::AKAZE::DescriptorType  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptor_size = 0; 
        int descriptor_channels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                    threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        //std::cout << "using SIFT descriptor" << std::endl;

        //SIFT descriptor implementation
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;

      extractor = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    } 
    else
    {
        //std::cout << "Given descriptor is not implemented." << std::endl;
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
   // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 2;      
    int kSize = 3;
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    cv::Mat corners;
    cv::cornerHarris(img, corners, blockSize, kSize, k, cv::BORDER_DEFAULT);
    cv::Mat corners_norm;
    cv::normalize(corners, corners_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );


    // add corners to result vector
    for (int j = 0; j < corners_norm.rows; j++) 
    {
        for (int i = 0; i < corners_norm.cols; i++) 
        {
            int response = (int)corners_norm.at<float>(j, i);
            if (response > 125) //thresholding on responses
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2*kSize;
                newKeyPoint.response = response;
                keypoints.push_back(newKeyPoint);
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat corners_scaled;
        cv::convertScaleAbs(corners_norm, corners_scaled);
        cv::Mat visImage = corners_scaled.clone();
        cv::drawKeypoints(corners_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsFast(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int threshold = 10;     
    bool nonmaxSuppression = true;
    cv::FastFeatureDetector::DetectorType  type = cv::FastFeatureDetector::TYPE_9_16;
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    
    //Create the fast feature detector
    cv::Ptr<cv::FastFeatureDetector> detector =  cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
    detector->detect(img, keypoints);



    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsBrisk(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
      // compute detector parameters based on image size
    int threshold = 30;     
    int octaves = 3;
    float patternScale = 1.0f;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    
    //Create the fast feature detector
    cv::Ptr<cv::BRISK> detector =  cv::BRISK::create(threshold, octaves, patternScale);
    detector->detect(img, keypoints);


    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }  
}

void detKeypointsOrb(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int nFeatures = 500;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    
    //Create the fast feature detector
    cv::Ptr<cv::ORB> detector =  cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, 
        WTA_K, scoreType, patchSize, fastThreshold);
    detector->detect(img, keypoints);


    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }  
}

void detKeypointsAkaze(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int descriptor_size = 0;
    int descriptor_channels = 3;
    float threshold = 0.001f;
    int nOctaves = 4;
    int nOctaveLayers = 4;
    cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        // Apply corner detection
    double t = (double)cv::getTickCount();
    
    //Create the fast feature detector
    cv::Ptr<cv::AKAZE> detector =  cv::AKAZE::create(descriptor_type, descriptor_size, 
        descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    detector->detect(img, keypoints);


    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }  
}

void detKeypointsSift(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold= 0.04;
    double edgeThreshold= 10;
    double sigma = 1.6;


    // Apply corner detection
    double t = (double)cv::getTickCount();
    
    //Create the fast feature detector
    cv::Ptr<cv::SIFT> detector =  cv::SIFT::create(nfeatures, nOctaveLayers, 
        contrastThreshold, edgeThreshold, sigma);
    detector->detect(img, keypoints);


    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }  
}