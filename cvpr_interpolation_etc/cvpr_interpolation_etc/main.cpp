
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
#include "FeatureMatching.h"
using namespace std;
using namespace cv;

clock_t timer;


#if 1
int main()
{
	int i = 7;
	char root[] = "E:/MyDocument/klive sync/计算机视觉/Stereo_Matching/Data Set/Middlebury2006/half size/data";
	char dirL[100];
	char dirR[100];
	char dirTrue[100];
	char dirDispC[100];
	char dirDispF[100];
	char dirDispS[100];
	char dirDispD[100];

for(int i=1;i<2;i++)
{
	sprintf(dirL,"%s%d%s",root,i,"/view1.png");
	sprintf(dirR,"%s%d%s",root,i,"/view5.png");
	sprintf(dirTrue,"%s%d%s",root,i,"/disp1.png");

	sprintf(dirDispC,"%s%d%s",root,i,"/DispC.png");
	sprintf(dirDispF,"%s%d%s",root,i,"/DispF.png");
	sprintf(dirDispS,"%s%d%s",root,i,"/DispS.png");
	sprintf(dirDispD,"%s%d%s",root,i,"/DispD.png");

	const cv::Mat imgL = cv::imread (dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	const cv::Mat trueDispImg = cv::imread (dirTrue, 0);

	double xDiffThresh = 12 ;
	double costThresh = 10;
	double peakRatio = 3;//cm2/cm1
	int winSize = 5;
	Mat sparseDisp(imgL.size(),CV_64FC1,Scalar(0));

	sparseDisparity(imgL,imgR,sparseDisp, xDiffThresh,costThresh, peakRatio, winSize);

	



	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 71;//双边滤波窗口大小
	cv::Mat FilterMat(sparseDisp.size(),CV_64FC1,Scalar(0));
	cv::Mat colorWeight(sparseDisp.size(),CV_64FC1,Scalar(0));//注意无效点处为0，否则会出错。
	timer = clock();
	//快速插值

	FastBFilterDepthInterp(sparseDisp, colorWeight, FilterMat, sparseDisp.cols, sparseDisp.rows, dSigma1, dSigma2, nWindows);//		BFilterDepthInterp(disp_coarse, colorWeight, FilterMat, disp_coarse.cols, disp_coarse.rows, dSigma1, dSigma2, nWindows);
	cout<<clock()-timer<<endl;	

	//深度求精
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	Mat isValidMap(imgL.size(),CV_8UC1,Scalar(0));
	int dLevels = 100;
	int winWidth = 7;
	int deviation = 10;
	double validThreshold = 10;
	double curvThreshold = -1.5;
	double peakRatioS = 1.2;

	//for temporary test
	//double M=0,m=1e10;
	//for(int i=0; i<sparseDisp.rows; i++)
	//{
	//	for(int j=0; j<sparseDisp.cols; j++)
	//	{
	//		if (sparseDisp.at<double>(i,j)> M) M = sparseDisp.at<double>(i,j);
	//		if (sparseDisp.at<double>(i,j)!=0 && sparseDisp.at<double>(i,j)<m ) m = sparseDisp.at<double>(i,j);
	//	}
	//}
	//vector<bool> isSampled(M+deviation+1);
	//vector<int > dispN(M+deviation+1);
	//for(int i = m;i<=M+deviation;i++) isSampled[i] = 1;
	//for(int i=0; i<sparseDisp.rows; i++)
	//{
	//	for(int j=0; j<sparseDisp.cols; j++)
	//	{
	//		 int disp = sparseDisp.at<double>(i,j);
	//		 if (disp!=0) isSampled[disp] = 1;
	//		//dispN[disp] ++;
	//	}
	//}
	vector<bool> isSampled(100);
	for(int i=0; i<isSampled.size();i++) 
	{
		if (i>30 && i<80 )
			isSampled[i] = 1;
		else
			isSampled[i] =0;
	}
	timer = clock();
	//深度求精函数
	boxFilterDepthRefine( imgL, imgR, FilterMat, isSampled, fineDepthMap,isValidMap, dLevels ,winWidth,deviation,validThreshold,curvThreshold,peakRatioS);
	cout<<clock()-timer<<endl;

	//for test
	Mat subtractImg(imgL.size(),CV_64FC1,Scalar(0));
	Mat diffImg(imgL.size(),CV_64FC1,Scalar(0));
	for(int i =0;i<fineDepthMap.rows;i++)
		for(int j=0;j<fineDepthMap.cols;j++)
		{
			subtractImg.at<double>(i,j) = (isValidMap.at<uchar>(i,j)==1 ? fineDepthMap.at<double>(i,j):0);
			diffImg.at<double>(i,j) = ( isValidMap.at<uchar>(i,j)==1 ? abs( 2* fineDepthMap.at<double>(i,j) - trueDispImg.at<uchar>(i,j) ):0 ); 
		}

	imwrite(dirDispC, FilterMat*2);//imwrite("data3/dispC.jpg",FilterMat);
	imwrite(dirDispF,fineDepthMap*2);//imwrite("data3/dispF.jpg",fineDepthMap);
	imwrite(dirDispS,subtractImg*2);//imwrite("data3/dispS.jpg",subtractImg);
	imwrite(dirDispD,diffImg);

	imshow("0",trueDispImg);
	imshow("1",FilterMat/60);
	imshow("2",sparseDisp/20);
	imshow("3",fineDepthMap/20);
	imshow("4",subtractImg/20);
	imshow("5",isValidMap*255);
	imshow("6",diffImg);
	waitKey(0);
}

}

#elif 1
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