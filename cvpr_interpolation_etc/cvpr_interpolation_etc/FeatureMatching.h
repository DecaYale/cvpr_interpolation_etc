#pragma once

#include "cxcore.h"

extern void sparseDisparity(const cv::Mat & imgL,const cv::Mat & imgR,cv::Mat & sparseDisp, double xDiffThresh, double costThresh, int winSize);