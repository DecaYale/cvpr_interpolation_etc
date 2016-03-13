#include "cxcore.h"
#include "time.h"
#include <iostream>
using namespace cv;

//构造offsetImg
void offsetGenerate(Mat& dispMap,Mat & offsetImg, int deviation)
{
	int width = dispMap.cols;
	int height = dispMap.rows;
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			int offset_t = dispMap.at<double>(y,x) - deviation;
			offsetImg.at<double>(y,x) = (offset_t) >=0 ? offset_t :0 ;
		}

	}
}
void integralImgCal(double * originImg, int height,int width,int widthStep)
{
	//first row
	double rs = 0.0f;
	int wStep = widthStep/sizeof(double);

	for(int j=0; j<width; j++)
	{
		rs += originImg[j];
		originImg[j] = rs;
	}
	//remaining rows
	for(int i=1; i<height; i++)
	{
		double rs = 0.0;
		for(int j=0; j<width; j++)
		{
			int idx = i*wStep + j;
			rs += originImg[idx];
			originImg[idx] = rs + originImg[(i-1)*wStep +j ];
		}
	}


}

void offsetGenerate2(cv::Mat& dispMap, cv::Mat & offsetImg, int deviation)
{
	int width = dispMap.cols;
	int height = dispMap.rows;
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			int offset_t = dispMap.at<double>(y,x) - deviation;
			offsetImg.at<double>(y,x) = (offset_t) >=0 ? offset_t :0 ;
		}

	}
}
//////////////////////////////////////////////////////////////////////////
//深度求精函数:boxFilterDepthRefine(...)
//input:
//		img_L,img_R:	双目左右图像,<uchar>类型
//		coarseDepthMap:	待求精的深度图,<double>类型
//		dLevels:		 最大的深度范围：根据实际情况调大，暂未优化
//		winWidth:		WTA local 采用的窗口大小
//		deviation:		在粗糙深度附近deviation内求精
//		validThreshold:	curvature判决阈值

//output:
//		fineDepthMap:	求精后的输出结果，<double>类型
//		isValid:		深度有效性图
//
//////////////////////////////////////////////////////////////////////////

void boxFilterDepthRefine(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat &coarseDepthMap, cv::Mat &fineDepthMap,cv:: Mat & isValid, int dLevels ,int winWidth,int deviation,double validThreshold,double curvThreshold)
{
	int height = img_L.rows;
	int width = img_L.cols;
	int patchSize=winWidth*winWidth;
	int *size = new int[3];
	size[0] = dLevels; size[1] = height; size[2] = width;
	double val = 1E5;	//大了会出问题，why？
	cv::Mat * dataCostCube = new Mat(3,size,CV_64FC1,Scalar(val) );
	cv::Mat * winCostCube = new Mat(3,size,CV_64FC1,Scalar(0) );
	cv::Mat * offsetMap = new Mat(img_L.size(),CV_64FC1,Scalar(0));

	offsetGenerate(coarseDepthMap, *offsetMap, deviation);
	assert(img_L.channels() == 1);
	//计算raw cost
	int range = 2*deviation+1;
	//for(int i=0;i<height;i++)
	//{
	//	for(int j=0; j<width; j++)
	//	{
	//		for(int d=offsetMap->at<double>(i,j);
	//				d < offsetMap->at<double>(i,j)+ 2*deviation+1;
	//				d++
	//			)
	//		{
	//			if (j-d < 0 || d>=dLevels) continue;

	//			dataCostCube->at<double>(d,i,j) = abs( 
	//													img_L.at<uchar>(i,j) - img_R.at<uchar>(i,j-d)
	//												);
	//		}
	//	}
	//}
#pragma omp parallel for
	for(int i=0;i<height;i++)
	{
		for(int j=0; j<width; j++)
		{
			for(int d=offsetMap->at<double>(i,j);
				d < offsetMap->at<double>(i,j)+ 2*deviation+1;
				d++
				)
			{
				if (j-d < 0 || d>=dLevels) continue;

				dataCostCube->at<double>(d,i,j) = abs( 
					img_L.at<uchar>(i,j) - img_R.at<uchar>(i,j-d)
					);
			}
		}
	}

	//计算summed area table
	Mat * costMap_d = new Mat(img_L.size(),CV_64FC1,Scalar(0));
#pragma omp parallel for
	for(int d=0; d<dLevels; d++)
	{
		double * costMap_d = (double*)(dataCostCube->data + d*( dataCostCube->step[0]) );
		integralImgCal(costMap_d, height,width,dataCostCube->step[1]);
	}


	//计算winCostCube
#pragma omp parallel for
	for(int d=0; d<dLevels; d++)
	{
		for(int i=0;i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				if (i-winWidth <0 || j-winWidth<0) continue;
				
				winCostCube->at<double>(d,i,j) = dataCostCube->at<double>(d,i,j)
					+dataCostCube->at<double>(d,i-winWidth,j-winWidth)
					-dataCostCube->at<double>(d,i-winWidth,j)
					-dataCostCube->at<double>(d,i,j-winWidth);

			}
		}

	}
	//WTA
#pragma omp parallel for
	for(int i=0;i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			double min_t = 1e20;
			double d_min = 0;
			double cost_tp ;
			double cost_tn ;
			double curvation;

			for(int d=0; d<dLevels; d++)
			{
				
				
				double cost_t = winCostCube->at<double>(d,i,j);//double cost_t = dataCostCube->at<double>(d,i,j);//

				//if (d==0)
				//{
				//	cost_tn = winCostCube->at<double>(d+1,i,j);
				//	cost_tp = cost_tn;
				//}
				//else if (d == dLevels-1)
				//{
				//	cost_tp = winCostCube->at<double>(d-1,i,j);
				//	cost_tn = cost_tp;
				//}
				//else 
				//{
				//	cost_tp = winCostCube->at<double>(d-1,i,j);
				//	cost_tn = winCostCube->at<double>(d+1,i,j);
				//}

				/*int n,m;
				int patchSize=winWidth*winWidth;
				m = (int)(cost_t/val);
				n = patchSize - m;
				cost_t = (cost_t - val* m)/n;

				m = (int)(cost_tn/val);
				n = patchSize - m;
				cost_tn = (cost_tn - val * m)/n;
				m = (int)(cost_tp/val);
				n = patchSize - m;
				cost_tp = (cost_tp - val * m)/n;*/

				/*for(n=winWidth*winWidth; cost_t>=val; cost_t-=val,n--);
				cost_t /= n;

				for(n=winWidth*winWidth; cost_tn>=val; cost_tn-=val,n--);
				cost_tn /= n;

				for(n=winWidth*winWidth; cost_tp>=val; cost_tp-=val,n--);
				cost_tp /= n;*/
				////更改
				//n = winWidth*winWidth;
				//winCostCube->at<double>(d,i,j)=cost_t*n;
				//if (d==0)
				//{
				//	winCostCube->at<double>(d+1,i,j) = cost_tn *n;
				//}
				//else if (d == dLevels-1)
				//{
				//	winCostCube->at<double>(d-1,i,j)=cost_tp*n;
				//}
				//else 
				//{
				//	winCostCube->at<double>(d-1,i,j)=cost_tp *n;
				//	winCostCube->at<double>(d+1,i,j) = cost_tn *n;
				//}


				if (cost_t < min_t ) 
				{
					min_t = cost_t;
					d_min = d;

					//curvation = 2*cost_t - cost_tn - cost_tp;
				}
			}
			fineDepthMap.at<double>(i,j) = d_min; //赋值为最小cost 的d

			//isValid Mat 赋值
			double cost_t = winCostCube->at<double>(d_min,i,j);

			if (d_min==0)
			{
				cost_tn = winCostCube->at<double>(d_min+1,i,j);
				cost_tp = cost_tn;
			}
			else if (d_min == dLevels-1)
			{
				cost_tp = winCostCube->at<double>(d_min-1,i,j);
				cost_tn = cost_tp;
			}
			else 
			{
				cost_tp = winCostCube->at<double>(d_min-1,i,j);
				cost_tn = winCostCube->at<double>(d_min+1,i,j);
			}
			int n,m;
			int patchSize=winWidth*winWidth;
			m = (int)(cost_t/val);
			n = patchSize - m;
			cost_t = (cost_t - val* m)/n;
			m = (int)(cost_tn/val);
			n = patchSize - m;
			cost_tn = (cost_tn - val * m)/n;
			m = (int)(cost_tp/val);
			n = patchSize - m;
			cost_tp = (cost_tp - val * m)/n;
			curvation = 2*cost_t - cost_tn - cost_tp;
			min_t = cost_t;//!!!
			if (min_t< validThreshold &&  curvation<curvThreshold)//if (min_t/(winWidth*winWidth - t) < validThreshold)
			{
				isValid.at<uchar>(i,j) = 1;
			}
			else 
			{
				isValid.at<uchar>(i,j) = 0;
			}

		}

	}

	////有效性确定
	//for(int i=0;i<height; i++)
	//{
	//	for(int j=0; j<width; j++)
	//	{
	//		int cost= abs( 
	//				img_L.at<uchar>(i,j) - img_R.at<uchar>(i, j- fineDepthMap.at<double>(i,j) )
	//				);

	//		if (cost<validThreshold) isValid.at<uchar>(i,j) = 1;
	//		else isValid.at<uchar>(i,j) = 0;
	//	}
	//}

	delete dataCostCube;
	delete winCostCube;
	delete offsetMap;
}



//void boxFilterDepthRefine(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat & coarseDepthMap,cv::Mat &fineDepthMap, cv::Mat offsetMap,int dLevels ,int winWidth)
//img_L和img_R 是单通道灰度uchar图像
void boxFilterDepthRefine_prototype(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat &coarseDepthMap, Mat &fineDepthMap, int dLevels ,int winWidth,int deviation)
{
	int height = img_L.rows;
	int width = img_L.cols;
	int *size = new int[3];
	size[0] = dLevels; size[1] = height; size[2] = width;
	double val = 1E5;	//大了会出问题，why？
	cv::Mat * dataCostCube = new Mat(3,size,CV_64FC1,Scalar(val) );
	cv::Mat * winCostCube = new Mat(3,size,CV_64FC1,Scalar(0) );
	cv::Mat * offsetMap = new Mat(img_L.size(),CV_64FC1,Scalar(0));

	offsetGenerate(coarseDepthMap, *offsetMap, deviation);
	assert(img_L.channels() == 1);
clock_t timer = clock();
	for(int i=0;i<height;i++)
	{
		for(int j=0; j<width; j++)
		{
			int d_ceil = offsetMap->at<double>(i,j)+ 2*deviation+1;
			for(int d=offsetMap->at<double>(i,j);
					d < d_ceil;
					d++
				)
			{
				if (j-d < 0 || d>=dLevels) continue;

				dataCostCube->at<double>(d,i,j) = abs( 
														img_L.at<uchar>(i,j) - img_R.at<uchar>(i,j-d)
													);
			}
		}
	}
std::cout<<clock()-timer<<std::endl;
timer = clock();
	//计算summed area table
	for(int d=0; d<dLevels; d++)
	{
		double * costMap_d = (double*)(dataCostCube->data + d*( dataCostCube->step[0]) );
		integralImgCal(costMap_d, height,width,dataCostCube->step[1]);
	}
std::cout<<clock()-timer<<std::endl;
timer = clock();
	//计算winCostCube
	for(int d=0; d<dLevels; d++)
	{
		for(int i=0;i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				if (i-winWidth <0 || j-winWidth<0) continue;

				winCostCube->at<double>(d,i,j) = dataCostCube->at<double>(d,i,j)
											+dataCostCube->at<double>(d,i-winWidth,j-winWidth)
											-dataCostCube->at<double>(d,i-winWidth,j)
											-dataCostCube->at<double>(d,i,j-winWidth);
			}
		}
		
	}
std::cout<<clock()-timer<<std::endl;
	//WTA
timer = clock();
	for(int i=0;i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			double min_t = 1e20;
			double d_min = 0;
			for(int d=0; d<dLevels; d++)
			{
				double cost_t = winCostCube->at<double>(d,i,j);//double cost_t = dataCostCube->at<double>(d,i,j);//
				if (cost_t < min_t ) 
				{
					min_t = cost_t;
					d_min = d;
				}
			}
			fineDepthMap.at<double>(i,j) = d_min; //赋值为最小cost 的d
		}

	}
std::cout<<clock()-timer<<std::endl;

}






//stereo matching using WTA through integral image algorithm
void boxFilterStereo(const cv::Mat/*<uchar>*/ & img_L,const cv::Mat/*<uchar>*/ & img_R,cv::Mat/*<double>*/ &fineDepthMap, int dLevels ,int winWidth)
{
	int height = img_L.rows;
	int width = img_L.cols;
	int *size = new int[3];
	size[0] = dLevels; size[1] = height; size[2] = width;
	double val = 1E5;	//大了会出问题，why？
	cv::Mat * dataCostCube = new Mat(3,size,CV_64FC1,Scalar(val) );
	cv::Mat * winCostCube = new Mat(3,size,CV_64FC1,Scalar(0) );

	assert(img_L.channels() == 1);

	for(int i=0;i<height;i++)
	{
		for(int j=0; j<width; j++)
		{
			for(int d=0;d<dLevels;d++)
			{
				if (j-d < 0) continue;

				dataCostCube->at<double>(d,i,j) = abs( 
					img_L.at<uchar>(i,j) - img_R.at<uchar>(i,j-d)
					);
			}
		}
	}

	//计算summed area table
	for(int d=0; d<dLevels; d++)
	{
		double * costMap_d = (double*)(dataCostCube->data + d*( dataCostCube->step[0]) );
		integralImgCal(costMap_d, height,width,dataCostCube->step[1]);
	}


	//计算winCostCube
	for(int d=0; d<dLevels; d++)
	{
		for(int i=0;i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				if (i-winWidth <0 || j-winWidth<0) continue;

				winCostCube->at<double>(d,i,j) = dataCostCube->at<double>(d,i,j)
					+dataCostCube->at<double>(d,i-winWidth,j-winWidth)
					-dataCostCube->at<double>(d,i-winWidth,j)
					-dataCostCube->at<double>(d,i,j-winWidth);
			}
		}

	}
	//WTA
	for(int i=0;i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			double min_t = 1e20;
			double d_min = 0;
			for(int d=0; d<dLevels; d++)
			{
				double cost_t = winCostCube->at<double>(d,i,j);//double cost_t = dataCostCube->at<double>(d,i,j);//
				if (cost_t < min_t ) 
				{
					min_t = cost_t;
					d_min = d;
				}
			}
			fineDepthMap.at<double>(i,j) = d_min; //赋值为最小cost 的d
		}

	}
}