
#include <iostream>
#include <stdio.h>
#include "cxcore.h"
#include <highgui.h>  
#include "cvaux.h"
#include "cxmisc.h"
#include "cv.h"   
#include "opencv2/nonfree/features2d.hpp"

#include "Bilateral Filter.h"
//#include "StereoMatcher.h"
#include "ExWRegularGridBP.h"
#include "AdditionalFuncs.h"

//#include "DBP.h"
#include "boxFilterDepthRefine.h"
using namespace std;
using namespace cv;

clock_t timer;

#if 1
//说明：
//主要是两个函数：深度插值函数:FastBFilterDepthInterp(...)和深度求精函数boxFilterDepthRefine（...）
//FastBFilterDepthInterp由稀疏深度图disp_coarse（无效处初始化为0）得到插值后的稠密深度FilterMat
//boxFilterDepthRefine 由在稠密深度FilterMat附近使用双目图像求精，得到求精后的深度结果fineDepthMap（结果不一定是正确的），isValidMap 中值为1的部分表示对应位置fineDepthMap较为可靠，0的位置舍弃。
int main()
{
	const cv::Mat imgL = cv::imread ("./data/scene1.row3.col3.ppm", 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread ("./data/scene1.row3.col4.ppm", 0);//("./data/scene1.row3.col4.ppm", 0);

	cv::Mat dispTrue = cv::imread ("./data/truedisp.row3.col3.pgm", 0);//("./data/truedisp.row3.col3.pgm", 0);//("./data2/disp1_half.png", 0);//("./data/truedisp.row3.col3.pgm", 0);
	dispTrue = dispTrue/15;///15;//处理后dispTrue是真是的深度数据

//sift 获得特征点并在相应深度图上采样获得disp_coarse
	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(imgL, keypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(imgL, keypoints, output);
	//cv::imwrite("sift_result.jpg", output);

	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//暂时先是CV_8UC1

//由稀疏深度插值得到稠密深度 
	for (int i=0;i<keypoints.size();i++)
	{
		disp_coarse.at<double>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x) = dispTrue.at<uchar>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x);
	}
	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 51;//双边滤波窗口大小
	cv::Mat FilterMat(disp_coarse.size(),CV_64FC1,Scalar(0));
	cv::Mat colorWeight(disp_coarse.size(),CV_64FC1,Scalar(0));//注意无效点处为0，否则会出错。
timer = clock();
	//快速插值
	FastBFilterDepthInterp(disp_coarse, colorWeight, FilterMat, disp_coarse.cols, disp_coarse.rows, dSigma1, dSigma2, nWindows);//		BFilterDepthInterp(disp_coarse, colorWeight, FilterMat, disp_coarse.cols, disp_coarse.rows, dSigma1, dSigma2, nWindows);
cout<<clock()-timer<<endl;	

//深度求精
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	Mat isValidMap(imgL.size(),CV_8UC1,Scalar(0));
	int dLevels = 20;
	int winWidth = 5;
	int deviation = 5;
	double validThreshold = 5;
	double curvThreshold = -2;
timer = clock();
	//深度求精函数
	boxFilterDepthRefine( imgL, imgR, FilterMat, fineDepthMap,isValidMap, dLevels ,winWidth,deviation,validThreshold,curvThreshold);
cout<<clock()-timer<<endl;

//for test
Mat subtractImg(imgL.size(),CV_64FC1,Scalar(0));
	for(int i =0;i<fineDepthMap.rows;i++)
		for(int j=0;j<fineDepthMap.cols;j++)
			subtractImg.at<double>(i,j) = (isValidMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j):0);

	imshow("0",dispTrue);//*255/15
	imshow("1",imgL);
	imshow("2",imgR);
	imshow("3",fineDepthMap/15);
	imshow("4",FilterMat/15);
	imshow("5",isValidMap*255);
	imshow("6",subtractImg/15);

	waitKey(0);
}


#endif