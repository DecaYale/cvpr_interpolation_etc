#pragma once

#include "cxcore.h"

extern void sparseDisparity(const cv::Mat & imgL,const cv::Mat & imgR,double xDiffThresh, int winSize);