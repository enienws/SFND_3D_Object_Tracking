
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f((left-250)/2.0, (bottom+50)/2.0), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f((left-250)/2.0, (bottom+125)/2.0), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::map<float, cv::DMatch> matchWithDistance;
    // Loop over all matches in the current frame
    float distAvr = 0.0;
    for (cv::DMatch match : kptMatches) 
    {    
        cv::Point2f kpPositionCurr = kptsCurr[match.trainIdx].pt;
        cv::Point2f kpPositionPrev = kptsPrev[match.queryIdx].pt;
        if (boundingBox.roi.contains(kpPositionCurr))
        {

            //boundingBox.kptMatches.push_back(match);
            float dist = sqrt((kpPositionCurr.x - kpPositionPrev.x) * (kpPositionCurr.x - kpPositionPrev.x) +
                                (kpPositionCurr.y - kpPositionPrev.y) * (kpPositionCurr.y - kpPositionPrev.y)); 
            matchWithDistance[dist] = match;
            distAvr += dist;
        }
    }
    //Calculate the average mean
    distAvr = distAvr / matchWithDistance.size();

    //Determine a adaptive threshold by using the average of all distances
    //20% percent more seems a good threshold for elimination
    float threshold = distAvr * 1.2;

    for(auto itr : matchWithDistance)
    {
        
        if(itr.first < threshold)
            boundingBox.kptMatches.push_back(itr.second);
    }

}


float calculateMeanDistance(std::vector<float> ratios)
{
        //Sort points by using the x axis, hence we get points from left to right
    std::sort(ratios.begin(), ratios.end());

    //Get a subset of lidar points to get rid of outliers
    int outlierRange = ratios.size() * 0.1;
    std::vector<float> filteredRatios(ratios.begin() + outlierRange, ratios.end() - outlierRange);

    float average = .0f;
    for(float item : filteredRatios)
    {
        average += item;
    }
    average = average / filteredRatios.size();

    return average;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    //vector for dk / dk' s calculated between the points
    std::vector<float> ratios;
    
    //Iterate between all possible matches for current and previous frames
    for(cv::DMatch matchOuter : kptMatches)
    {
        cv::KeyPoint outerKpCurr = kptsCurr[matchOuter.trainIdx];
        cv::KeyPoint outerKpPrev = kptsPrev[matchOuter.queryIdx];

        for(cv::DMatch matchInner : kptMatches)
        {
            cv::KeyPoint innerKpCurr = kptsCurr[matchInner.trainIdx];
            cv::KeyPoint innerKpPrev = kptsPrev[matchInner.queryIdx];

            //Calculate the distance between the pairs both for previous and current frames
            float distCurr = cv::norm(outerKpCurr.pt - innerKpCurr.pt);
            float distPrev = cv::norm(outerKpPrev.pt - innerKpPrev.pt);

            
            if(distPrev > std::numeric_limits<float>::epsilon())
            {
                float ratio = distCurr / distPrev;
                if(!ratio < std::numeric_limits<float>::epsilon())
                    ratios.push_back(ratio);
            }
                
        }
    }

    //Calculate the mean from the ratios
    float meanRatio = calculateMeanDistance(ratios);

    //Calculate the TTC
    TTC = (-1.0 / frameRate) / (1 - meanRatio);
}

// Helper function to sort lidar points based on their X (longitudinal) coordinate
float calculateLidarDistance(std::vector<LidarPoint> &lidarPoints)
{
    //Sort points by using the x axis, hence we get points from left to right
    std::sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint a, LidarPoint b) {
        return a.x < b.x;  
    });

    //Get a subset of lidar points to get rid of outliers
    int outlierRange = lidarPoints.size() * 0.01;
    std::vector<LidarPoint> filteredLidarPoints(lidarPoints.begin() + outlierRange, lidarPoints.end() - outlierRange);
    //std::cout << "filtered size: " << filteredLidarPoints.size() << std::endl;
    //Calculate average distance of filtered points
    float distance = 0.0;
    for(auto lidarPoint : filteredLidarPoints)
    {
        distance += lidarPoint.x;
    }
    distance = distance / filteredLidarPoints.size();
    
    return distance;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    //Get distance for previous frame
    float d0 = calculateLidarDistance(lidarPointsPrev);

    //Get distance for current frame
    float d1 = calculateLidarDistance(lidarPointsCurr);

    //Calculate TTC 
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //Matches mmap, currBoxID -> prevBoxID
    std::multimap<int, int> matchVote;

    //Iterate over all matches
    for(int i=0; i<matches.size(); i++)
    {
        cv::DMatch & currMatch = matches[i];
        cv::KeyPoint & prevKp = prevFrame.keypoints[currMatch.queryIdx];
        cv::KeyPoint & currKp = currFrame.keypoints[currMatch.trainIdx];

        int prevBBoxID = -1;        
        //Get the bbox id for the previous frames's matching kp
        for(BoundingBox prevBBox : prevFrame.boundingBoxes)
        {
            if(prevBBox.roi.contains(prevKp.pt))
            {
                prevBBoxID = prevBBox.boxID;
            }
        }

        int currBBoxID = -1;        
        //Get the bbox id for the previous frames's matching kp
        for(BoundingBox currBBox : currFrame.boundingBoxes)
        {
            if(currBBox.roi.contains(currKp.pt))
            {
                currBBoxID = currBBox.boxID;
            }
        }

        //Create a match
        if(prevBBoxID == -1 || currBBoxID == -1)
            continue;
        else
            matchVote.insert({currBBoxID, prevBBoxID});
    }

    //Sum up the votes
    for(BoundingBox currBBox : currFrame.boundingBoxes)
    {
        
        //prevBoxId, matchCount
        std::map<int, int> voteForCurrBox;
        int id = currBBox.boxID;

        auto range = matchVote.equal_range(id);
        for (auto vote = range.first; vote != range.second; vote++)
        {
            int matchingPrevBoxID = vote->second;
            if(voteForCurrBox.find(matchingPrevBoxID) == voteForCurrBox.end())
            {
                //Not found insert a new element
                voteForCurrBox[matchingPrevBoxID] = 1;
            }
            else
            {
                //Update the vote count
                int vote = voteForCurrBox[matchingPrevBoxID];
                voteForCurrBox[matchingPrevBoxID] = ++vote;
            }
        }

        //Create a match with largest vote counts

        unsigned maxVal = 0;
        unsigned maxPrevBoxID = 0;
        for(auto it = voteForCurrBox.cbegin(); it != voteForCurrBox.cend(); ++it ) 
        {
            if (it ->second > maxVal) 
            {
                maxPrevBoxID = it->first;
                maxVal = it->second;
            }
        }

        //Insert the found match to the map
        //std::cout << "Add a match: " << maxPrevBoxID << " -> " << id << std::endl;
        bbBestMatches[maxPrevBoxID] = id;
    }
}
