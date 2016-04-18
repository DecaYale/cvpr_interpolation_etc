
#include "LocalStereoMatching.h"
#include "cxcore.h"
#include "Hebf/qx_hardware_efficient_bilateral_filter.h"
#include "Hebf/qx_basic.h"
using namespace cv;
void CLocalStereoMatching::offsetGenerate(const Mat& dispMap,Mat & offsetImg)
{
	int width = dispMap.cols;
	int height = dispMap.rows;
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			int offset_t = dispMap.at<double>(y,x) - m_deviation;
			offsetImg.at<double>(y,x) = (offset_t) >=0 ? offset_t :0 ;
		}

	}
}
void CLocalStereoMatching::integralImgCal(double * originImg, int height,int width,int widthStep)
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
void CLocalStereoMatching::boxFilterDepthRefine(const cv::Mat &coarseDepthMap,cv:: Mat & fineDepthMap)
{

	int height = imgL->rows;
	int width = imgL->cols;
	int patchArea=m_winWidth*m_winWidth;//int patchSize=winWidth*winWidth;
	int *size = new int[3];

	//dLevels = dispSample.size();//+1+1;//dlevels 是图像中最大的disparity valued
	size[0] = m_dLevels; size[1] = height; size[2] = width;
	double val = 1E5;	//大了会出问题，why？
	m_dataCostCube = new Mat(3,size,CV_64FC1,Scalar(val) );
	m_dataCostCubeIntegral = new Mat(3,size,CV_64FC1,Scalar(val) );//m_dataCostCubeIntegral = new Mat();//

	m_winCostCube = new Mat(3,size,CV_64FC1,Scalar(0) );
	cv::Mat * offsetMap = new Mat(imgL->size(),CV_64FC1,Scalar(0));

	////offsetGenerate(coarseDepthMap, *offsetMap);
	assert(imgL->channels() == 1);
	//计算raw cost
	int range = 2*m_deviation+1;
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
	for(int i=0;i<height;i++)
	{
		for(int j=0; j<width; j++)
		{
			for(int d= 0;//offsetMap->at<double>(i,j);
				d < m_dLevels;//offsetMap->at<double>(i,j)+ 2*m_deviation+1;
				d++
				)
			{
				//if (j-d < 0 || d>=m_dLevels) continue;
				int jr = max(j-d,0);//if (j-d < 0 || d>=m_dLevels) continue;
				if (d>=m_dLevels) continue;//d = min(d,m_dLevels-1);

				m_dataCostCube->at<double>(d,i,j) = abs( 
					imgL->at<uchar>(i,j) - imgR->at<uchar>(i,jr)//imgL->at<uchar>(i,j) - imgR->at<uchar>(i,j-d)
					);
			}
		}
	}


	////计算summed area 
	 * m_dataCostCubeIntegral =  m_dataCostCube->clone(); 
	Mat * costMap_d = new Mat(imgL->size(),CV_64FC1,Scalar(0));
	for(int d=0; d<m_dLevels; d++)
	{
		//if ( !dispSample.at(d) ) continue;///

		double * costMap_d = (double*)(m_dataCostCubeIntegral->data + d*( m_dataCostCubeIntegral->step[0]) );
		integralImgCal(costMap_d, height,width,m_dataCostCubeIntegral->step[1]);
	}



	//计算winCostCube
	for(int d=0; d<m_dLevels; d++)
	{
		//if (! dispSample.at(d)) continue;///

		for(int i=0;i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				if (i-m_winWidth <0 || j-m_winWidth<0) continue;
				
				m_winCostCube->at<double>(d,i,j) = (
					m_dataCostCubeIntegral->at<double>(d,i,j)
					+m_dataCostCubeIntegral->at<double>(d,i-m_winWidth,j-m_winWidth)
					-m_dataCostCubeIntegral->at<double>(d,i-m_winWidth,j)
					-m_dataCostCubeIntegral->at<double>(d,i,j-m_winWidth)
					); // /patchArea 取平均

			}
		}

	}
//	//qx hebf
//	for (int d =0; d<m_dLevels; d++)
//	{
//		unsigned char *** image,***texture,*** image_filtered;
//		//double ***image_filtered;
//		image_filtered = qx_allocu_3(height,width,3);
//		image = qx_allocu_3(height,width,3);
//		texture = qx_allocu_3(height,width,3);
////clock_t timer = clock();
//		for(int i=0; i<height; i++)
//		{
//			for(int j=0;j<width; j++)
//			{
//				image[i][j][0] = m_dataCostCube->at<double>(d,i,j);
//				image[i][j][1] = m_dataCostCube->at<double>(d,i,j);
//				image[i][j][2] = m_dataCostCube->at<double>(d,i,j);
//
//				texture[i][j][0] = imgL->at<uchar>(i,j);
//				texture[i][j][1] = texture[i][j][0];
//				texture[i][j][2] = texture[i][j][0];
//					
//			}
//		}
////cout<< clock()-timer<<endl;
//		int scale = 3;int sigma = 10; int radius=2;
//		qx_hardware_efficient_bilateral_filter m_hebf;
//		m_hebf.init(height,width,3,scale,sigma,radius);//initialization
//		m_hebf.filter(image_filtered,image,texture);//m_hebf.filter(image_filtered,image,image);//bilateral filtering
//
//		for(int i=0; i<height; i++)
//		{
//			for(int j=0;j<width; j++)
//			{
//				m_winCostCube->at<double>(d,i,j) = image_filtered[i][j][0] ;
//			}
//		}
//
//	}
//
	//WTA
	for(int i=0;i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			double min_t = 1e20;
			double min2_t = 1e20;
			double d_min = 0;
			double d_min2 = 0;
			double cost_tp ;
			double cost_tn ;
			double curvation;

			for(int d=0; d<m_dLevels; d++)
			{
				//if (! dispSample[d]) continue;///
				
				double cost_t = m_winCostCube->at<double>(d,i,j);//double cost_t = dataCostCube->at<double>(d,i,j);//

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
					min2_t = min_t; //次小的
					d_min2 = d_min;

					min_t = cost_t; // 最小的
					d_min = d;

					//curvation = 2*cost_t - cost_tn - cost_tp;
				}
				else if (cost_t < min2_t)
				{
					min2_t  = cost_t;
					d_min2 = d;
				}
			}
			fineDepthMap.at<double>(i,j) = d_min; //赋值为最小cost 的d

			//isValid Mat 赋值
			double cost_t = m_winCostCube->at<double>(d_min,i,j);
			double cost2_t = m_winCostCube->at<double>(d_min2,i,j);

			if (d_min==0)
			{
				cost_tn = m_winCostCube->at<double>(d_min+1,i,j);
				cost_tp = cost_tn;
			}
			else if (d_min == m_dLevels-1)
			{
				cost_tp = m_winCostCube->at<double>(d_min-1,i,j);
				cost_tn = cost_tp;
			}
			else 
			{
				cost_tp = m_winCostCube->at<double>(d_min-1,i,j);
				cost_tn = m_winCostCube->at<double>(d_min+1,i,j);
			}
			//提取实际cost

			int n,m;
			//int patchSize=winWidth*winWidth;
			m = (int)(cost_t/val);
			n = patchArea - m;
			cost_t = (cost_t - val* m)/n;

			m = (int)(cost2_t/val);
			n = patchArea - m;
			cost2_t = (cost2_t - val* m)/n;

			m = (int)(cost_tn/val);
			n = patchArea - m;
			cost_tn = (cost_tn - val * m)/n;
			m = (int)(cost_tp/val);
			n = patchArea - m;
			cost_tp = (cost_tp - val * m)/n;
			curvation = 2*cost_t - cost_tn - cost_tp;
			min_t = cost_t;//!!!
			if (min_t< m_validThreshold &&  curvation<m_curvThreshold && cost2_t/cost_t > m_peakRatio)//if (min_t/(winWidth*winWidth - t) < validThreshold)
			{
				m_confidenceMap->at<uchar>(i,j) = 1;
			}
			else 
			{
				m_confidenceMap->at<uchar>(i,j) = 0;
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

	//delete dataCostCube;
	//delete winCostCube;
	delete offsetMap;


}

void CLocalStereoMatching:: getDataCostCube(cv::Mat & data_cost_cube)
{
	 data_cost_cube = m_dataCostCube->clone();
}

void CLocalStereoMatching:: getConfidenceMap(cv::Mat & confidence_map)
{
	confidence_map = m_confidenceMap->clone();
}