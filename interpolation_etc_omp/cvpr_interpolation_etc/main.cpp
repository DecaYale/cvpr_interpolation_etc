
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
//˵����
//��Ҫ��������������Ȳ�ֵ����:FastBFilterDepthInterp(...)������󾫺���boxFilterDepthRefine��...��
//FastBFilterDepthInterp��ϡ�����ͼdisp_coarse����Ч����ʼ��Ϊ0���õ���ֵ��ĳ������FilterMat
//boxFilterDepthRefine ���ڳ������FilterMat����ʹ��˫Ŀͼ���󾫣��õ��󾫺����Ƚ��fineDepthMap�������һ������ȷ�ģ���isValidMap ��ֵΪ1�Ĳ��ֱ�ʾ��Ӧλ��fineDepthMap��Ϊ�ɿ���0��λ��������
int main()
{
	const cv::Mat imgL = cv::imread ("./data/scene1.row3.col3.ppm", 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread ("./data/scene1.row3.col4.ppm", 0);//("./data/scene1.row3.col4.ppm", 0);

	cv::Mat dispTrue = cv::imread ("./data/truedisp.row3.col3.pgm", 0);//("./data/truedisp.row3.col3.pgm", 0);//("./data2/disp1_half.png", 0);//("./data/truedisp.row3.col3.pgm", 0);
	dispTrue = dispTrue/15;///15;//�����dispTrue�����ǵ��������

//sift ��������㲢����Ӧ���ͼ�ϲ������disp_coarse
	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(imgL, keypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(imgL, keypoints, output);
	//cv::imwrite("sift_result.jpg", output);

	cv::Mat disp_coarse(imgL.size(),CV_64FC1,Scalar(0));//��ʱ����CV_8UC1

//��ϡ����Ȳ�ֵ�õ�������� 
	for (int i=0;i<keypoints.size();i++)
	{
		disp_coarse.at<double>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x) = dispTrue.at<uchar>((int)keypoints.at(i).pt.y,(int) keypoints.at(i).pt.x);
	}
	double dSigma1 = 10;
	double dSigma2 = 100;
	int nWindows = 51;//˫���˲����ڴ�С
	cv::Mat FilterMat(disp_coarse.size(),CV_64FC1,Scalar(0));
	cv::Mat colorWeight(disp_coarse.size(),CV_64FC1,Scalar(0));//ע����Ч�㴦Ϊ0����������
timer = clock();
	//���ٲ�ֵ
	FastBFilterDepthInterp(disp_coarse, colorWeight, FilterMat, disp_coarse.cols, disp_coarse.rows, dSigma1, dSigma2, nWindows);//		BFilterDepthInterp(disp_coarse, colorWeight, FilterMat, disp_coarse.cols, disp_coarse.rows, dSigma1, dSigma2, nWindows);
cout<<clock()-timer<<endl;	

//�����
	Mat fineDepthMap(imgL.size(),CV_64FC1,Scalar(0));
	Mat isValidMap(imgL.size(),CV_8UC1,Scalar(0));
	int dLevels = 20;
	int winWidth = 5;
	int deviation = 5;
	double validThreshold = 5;
	double curvThreshold = -2;
timer = clock();
	//����󾫺���
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