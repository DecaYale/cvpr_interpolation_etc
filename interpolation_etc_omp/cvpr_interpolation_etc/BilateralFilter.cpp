
#include "cxcore.h"
#include "time.h"
#include <iostream>
using namespace std;
using namespace cv;

/////////////////////////////////////////////////////////////////////////////////////
//函数说明:用于快速深度插值的双边滤波函数
//input:
//		pGrayMat	稀疏深度图,<double>类型
//		colorWeight:		<double>类型
//		nWidth:		图像宽度
//		nHeight:	图像高度
//		dSigma1、dSigma2:分别为几何与灰度相关高斯函数的方差		
//		nWinWidth	:双边滤波窗口大小
//output:
//		FilterMat:	求精后的输出结果，<double>类型
//		
//
//////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
void FastBFilterDepthInterp(Mat &pGrayMat, Mat& colorWeight,Mat & FilterMat, int nWidth, int nHeight, double dSigma1, double dSigma2, int nWinWidth,double INVAILD)  
{
	////////////////////////参数说明///////////////////////////////////  
	//pGrayMat:待处理图像数组  
	//pFilterMat:保存高斯滤波结果  
	//nWidth:图像宽度  
	//nHeight:图像高度  
	//dSigma1、dSigma2:分别为几何与灰度相关高斯函数的方差  
	const double e = 2.718281828459;
	double * distWeight = new double[nWinWidth];

	const double BETA = exp(-1/dSigma1);//const double BETA = exp(-0.5/dSigma1);
	const double BETA_INV = exp(1/dSigma1);//const double BETA_INV = exp(0.5/dSigma1);
	for(int i=0; i<nWinWidth; i++)  
	{  
		distWeight[i] = abs(i-nWinWidth/2);//((nWindows-1)/2-nNumX)*((nWindows-1)/2-nNumX) + ((nWindows-1)/2-nNumY)*((nWindows-1)/2-nNumY);  
		distWeight[i] = exp(-distWeight[i]/dSigma1);//exp(-0.5*distWeight[i]/dSigma1);   //C参数  

		//static FILE * fout = fopen("tmp3.txt","w+");
		//fprintf(fout,"%f %f \n",distWeight[i] );
	}  

	for(int i=0;i<nHeight;i++)
	{
		for(int j=0;j<nWidth;j++)
		{
			double dGray = pGrayMat.at<double>(i,j);
			//if (dGray != INVAILD)
				colorWeight.at<double>(i,j) = 1;//exp(-0.5*dGray*dGray/dSigma2/dSigma2); //S参数  
			//else 
				//colorWeight.at<double>(i,j) = 0;

		}
	}
//计算窗口aggregation
	cv::Mat tmp_hor_l(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_hor_r(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_ver_u(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_ver_d(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_hor(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_hor_l(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_hor_r(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_ver_u(colorWeight.size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_ver_d(colorWeight.size(),CV_64FC1,Scalar(0));
	//计算 tmp_hor_l
#pragma omp parallel for
	for(int i=0; i<nHeight; i++)
	{
		double sum = 0;
		int hor_l_cnt = 0;

		for(int j0=0; j0<nWinWidth/2+1; j0++)
		{
			if (pGrayMat.at<double>(i,j0) != 0) //注意！此处认为pGrayMat 值为0 是无效的
			{
				hor_l_cnt++;
				total_wght_hor_l.at<double>(i,nWinWidth/2)+=colorWeight.at<double>(i,j0) //s1
													*distWeight[j0];
			}

			sum += colorWeight.at<double>(i,j0)
					*distWeight[j0]
					*pGrayMat.at<double>(i,j0);

			
		}

		tmp_hor_l.at<double>(i,nWinWidth/2) = sum;///(0.001+total_wght_hor_l.at<double>(i,nWinWidth/2));//s2 //sum / (1e-3+ hor_l_cnt); //sum/(nWinWidth/2+1);//第一个有效赋值
		for(int j=0;j<nWidth; j++)
		{
			//注意边界
			if (j-nWinWidth/2-1 <0)continue;
			//tmp_hor_l.at<double>(i,j) = sum/(nWinWidth/2+1);
			
			//更新窗口内有效像素点数
			total_wght_hor_l.at<double>(i,j) = total_wght_hor_l.at<double>(i,j-1);//s8 赋初值
			if (pGrayMat.at<double>(i,j-nWinWidth/2-1) != 0)
			{
					hor_l_cnt--;
					total_wght_hor_l.at<double>(i,j) -= colorWeight.at<double>(i,j-nWinWidth/2-1) //s3
														*distWeight[0];
			}
			total_wght_hor_l.at<double>(i,j) *= BETA;//s7
			if (pGrayMat.at<double>(i,j) != 0) 
			{
					hor_l_cnt++;
					total_wght_hor_l.at<double>(i,j) += colorWeight.at<double>(i,j) //s4
															*distWeight[nWinWidth/2];
			}
			
			sum = (
					sum - colorWeight.at<double>(i,j-nWinWidth/2-1)* distWeight[0] * pGrayMat.at<double>(i,j-nWinWidth/2-1)
				   )*BETA
				   + colorWeight.at<double>(i,j)*  distWeight[nWinWidth/2] * pGrayMat.at<double>(i,j);

			if (hor_l_cnt == 0) total_wght_hor_l.at<double>(i,j) = 0;//s6 强制约束，防止浮点精度问题
			tmp_hor_l.at<double>(i,j) = sum;/// (0.001+total_wght_hor_l.at<double>(i,j)); //s5 //sum/ (0.001+hor_l_cnt);//sum/(1e-6+ hor_l_cnt);///sum/(nWinWidth/2+1);
			//if (hor_l_cnt!=0 && abs(total_wght_hor_l.at<double>(i,j))<0.001)
			//{
			//	static FILE * fout = fopen("tmpHorL.txt","w+");///////////////////////////
			//	fprintf(fout,"%d %d\n",i,j);//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_hor_l.at<double>(i,j),tmp_hor_l.at<double>(i,j) ,hor_l_cnt);////////////////
			//}
			//static FILE * fout = fopen("tmpHorL.txt","w+");///////////////////////////
			//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_hor_l.at<double>(i,j),tmp_hor_l.at<double>(i,j) ,hor_l_cnt);////////////////



			assert( (abs(sum)<0.001&&hor_l_cnt==0)||(abs(sum)>0.001&&hor_l_cnt!=0) );
		}

	}
	//////计算 tmp_hor_r //正向好像有浮点精度问题
	//for(int i=0; i<nHeight; i++)
	//{
	//	int sum = 0;
	//	int hor_r_cnt = 0;
	//	for(int j0=1;j0<nWinWidth/2+1;j0++)//for(int j0=nWinWidth/2+1; j0<nWinWidth; j0++)
	//	{
	//		if (pGrayMat.at<uchar>(i,j0) != 0) hor_r_cnt++;

	//		sum += colorWeight.at<double>(i,j0)
	//			*distWeight[j0+nWinWidth/2+1]
	//			*pGrayMat.at<uchar>(i,j0);
	//	}
	//	tmp_hor_r.at<double>(i,0) = sum/(0.001+hor_r_cnt);
	//	for(int j=1;j<nWidth; j++)
	//	{
	//		//注意边界
	//		if (j+nWinWidth/2 >= nWidth)continue;
	//		//tmp_hor_r.at<double>(i,j) = sum;//取平均/(nWinWidth/2)
	//		//更新窗口内有效像素点数
	//		if (pGrayMat.at<uchar>(i,j) != 0) hor_r_cnt--;
	//		if (pGrayMat.at<uchar>(i,j+nWinWidth/2) != 0) hor_r_cnt++;

	//		static FILE * fout = fopen("tmp.txt","w+");
	//		fprintf(fout,"%f %f \n",(sum - colorWeight.at<double>(i,j)* distWeight[nWinWidth/2+1] * pGrayMat.at<uchar>(i,j))*BETA_INV,sum );
	//		sum = (
	//			sum - colorWeight.at<double>(i,j)* distWeight[nWinWidth/2+1] * pGrayMat.at<uchar>(i,j)
	//			)*BETA_INV
	//			+ colorWeight.at<double>(i,j+nWinWidth/2)*  distWeight[nWinWidth-1] * pGrayMat.at<uchar>(i,j+nWinWidth/2);
	//		tmp_hor_r.at<double>(i,j) = sum/(0.001+hor_r_cnt);
	//	}

	//}

	///计算 tmp_hor_r //反向计算
#pragma omp parallel for
	for(int i=0; i<nHeight; i++)
	{
		double sum = 0;
		int hor_r_cnt = 0;
		for(int j0=nWidth-1, k=nWinWidth-1;j0 > nWidth-1-nWinWidth/2;j0--,k--)//for(int j0=nWinWidth/2+1; j0<nWinWidth; j0++)
		{
			if (pGrayMat.at<double>(i,j0) != 0)
			{
					hor_r_cnt++;
					total_wght_hor_r.at<double>(i,nWidth-1-nWinWidth/2)+=colorWeight.at<double>(i,j0) //s1
																		*distWeight[k];//*distWeight[j0];
			}

			sum += colorWeight.at<double>(i,j0)
				*distWeight[k]
				*pGrayMat.at<double>(i,j0);
		}
		tmp_hor_r.at<double>(i,nWidth-1-nWinWidth/2) = sum;///(0.001 + total_wght_hor_r.at<double>(i,nWidth-1-nWinWidth/2));//s2 //tmp_hor_r.at<double>(i,nWidth-1-nWinWidth/2) = sum /(0.001+hor_r_cnt);
		for(int j=nWidth-1;j>=0; j--)
		{
			//注意边界
			if (j+nWinWidth/2+1 >= nWidth || nWinWidth ==1 )continue; //nWinWidth ==1 则跳过不计算

			//tmp_hor_r.at<double>(i,j) = sum;//取平均/(nWinWidth/2)
			//更新窗口内有效像素点数
			total_wght_hor_r.at<double>(i,j) = total_wght_hor_r.at<double>(i,j+1);//s8

			if (pGrayMat.at<double>(i,j+nWinWidth/2+1) != 0) 
			{
				hor_r_cnt--;
				total_wght_hor_r.at<double>(i,j) -= colorWeight.at<double>(i,j+nWinWidth/2+1) //s3
													*distWeight[nWinWidth-1];
			}
			total_wght_hor_r.at<double>(i,j) *= BETA;//s7
			if ( pGrayMat.at<double>(i,j+1) != 0)
			{
				hor_r_cnt++;
				total_wght_hor_r.at<double>(i,j) += colorWeight.at<double>(i,j+1) //s4
					*distWeight[nWinWidth/2+1];
			}

			sum = (
				sum - colorWeight.at<double>(i,j+nWinWidth/2+1)* distWeight[nWinWidth-1] * pGrayMat.at<double>(i,j+nWinWidth/2+1)
				)*BETA
				+ colorWeight.at<double>(i,j+1)*  distWeight[nWinWidth/2+1] * pGrayMat.at<double>(i,j+1);

			if (hor_r_cnt == 0) total_wght_hor_r.at<double>(i,j) = 0;//s6 强制约束，防止浮点精度问题
			tmp_hor_r.at<double>(i,j) = sum;/// (0.001+total_wght_hor_r.at<double>(i,j)); //s5 //tmp_hor_r.at<double>(i,j) = sum/(0.001+hor_r_cnt);
			//static FILE * fout = fopen("tmpHorR.txt","w+");
			//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_hor_r.at<double>(i,j),tmp_hor_r.at<double>(i,j) ,hor_r_cnt);
			assert( (abs(sum)<0.001&&hor_r_cnt==0)||(abs(sum)>0.001&&hor_r_cnt!=0) );
		}

	}
	//...


	//计算融合以上2个矩阵的 FilterMat
#pragma omp parallel for
	for(int i=0; i<nHeight; i++)
	{
		for(int j=0; j<nWidth; j++)
		{
			//tmp_hor_l.at<double>(i,j) /= ( 0.001+total_wght_hor_l.at<double>(i,j) );
			double w_total = ( 0.001+total_wght_hor_l.at<double>(i,j) + total_wght_hor_r.at<double>(i,j) );
			tmp_hor.at<double>(i,j) =  (tmp_hor_l.at<double>(i,j) + tmp_hor_r.at<double>(i,j) ) / w_total;//(tmp_hor_l.at<double>(i,j) + tmp_hor_r.at<double>(i,j))*0.5;//FilterMat.at<uchar>(i,j) =  (tmp_hor_l.at<double>(i,j) + tmp_hor_r.at<double>(i,j))*0.5;
		}
	}

	// 纵向计算
	//计算 tmp_ver_u
#pragma omp parallel for
	for(int j=0; j<nWidth; j++)//for(int i=0; i<nHeight; i++)
	{
		double sum = 0;
		int ver_u_cnt = 0;

		for(int i0=0; i0<nWinWidth/2+1; i0++)//for(int j0=0; j0<nWinWidth/2+1; j0++)
		{
			if (abs( tmp_hor.at<double>(i0,j) )>0.001 ) 
			{
				ver_u_cnt++;
				total_wght_ver_u.at<double>(nWinWidth/2,j)+=colorWeight.at<double>(i0,j) //s1
					*distWeight[i0];
			}
			sum += colorWeight.at<double>(i0,j)
				*distWeight[i0]
			*tmp_hor.at<double>(i0,j);
		}
		assert( (abs(sum)<0.001&&ver_u_cnt==0)||(abs(sum)>0.001&&ver_u_cnt!=0) );

		tmp_ver_u.at<double>(nWinWidth/2,j) = sum;///(0.001+total_wght_ver_u.at<double>(nWinWidth/2,j));//s2 //tmp_ver_u.at<double>(nWinWidth/2, j) = sum / (1e-3+ ver_u_cnt); //sum/(nWinWidth/2+1);//第一个有效赋值
		for(int i=0;i<nHeight; i++)//for(int j=0;j<nWidth; j++)
		{
			//注意边界
			if (i-nWinWidth/2-1 <0)continue;
			//tmp_hor_l.at<double>(i,j) = sum/(nWinWidth/2+1);
			
			//更新窗口内有效像素点数
			total_wght_ver_u.at<double>(i,j) = total_wght_ver_u.at<double>(i-1,j);//s8
			if (abs( tmp_hor.at<double>(i-nWinWidth/2-1,j) )>0.001) 
			{
				ver_u_cnt--;
				total_wght_ver_u.at<double>(i,j) -= colorWeight.at<double>(i-nWinWidth/2-1,j) //s3
					*distWeight[0];
			}
			total_wght_ver_u.at<double>(i,j) *= BETA;//s7
			if (abs( tmp_hor.at<double>(i,j) ) >0.001) 
			{
				ver_u_cnt++;
				total_wght_ver_u.at<double>(i,j) += colorWeight.at<double>(i,j) //s4
					*distWeight[nWinWidth/2];
			}
			sum = (
				sum - colorWeight.at<double>(i-nWinWidth/2-1,j)* distWeight[0] * tmp_hor.at<double>(i-nWinWidth/2-1,j)
				)*BETA
				+ colorWeight.at<double>(i,j)*  distWeight[nWinWidth/2] * tmp_hor.at<double>(i,j);

			if (ver_u_cnt == 0) total_wght_ver_u.at<double>(i,j) = 0;//s6 强制约束，防止浮点精度问题
			tmp_ver_u.at<double>(i,j) = sum;/// (0.001+total_wght_ver_u.at<double>(i,j)); //s5 //tmp_ver_u.at<double>(i,j) = sum/ (0.001+ver_u_cnt);//sum/(1e-6+ hor_l_cnt);///sum/(nWinWidth/2+1);
			//cout<<i<<' '<<j<<' '<<sum<<' '<<ver_u_cnt<<endl;
			//static FILE * fout = fopen("tmp2.txt","w+");///////////////////////////
			//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_ver_u.at<double>(i,j),tmp_ver_u.at<double>(i,j) ,ver_u_cnt );////////////////
			assert( (abs(sum)<0.001&&ver_u_cnt==0)||(abs(sum)>0.001&&ver_u_cnt!=0) );
		}

	}
#pragma omp parallel for
	//计算 tmp_ver_d
	for(int j=0; j<nWidth; j++)//for(int i=0; i<nHeight; i++)
	{
		double sum = 0;
		int cnt = 0;
		for(int i0=nHeight-1, k=nWinWidth-1;i0 > nHeight-1-nWinWidth/2;i0--,k--)//for(int j0=nWinWidth/2+1; j0<nWinWidth; j0++)
		{
			if ( abs(tmp_hor.at<double>(i0,j) ) > 0.001)
			{
				cnt++;
				total_wght_ver_d.at<double>(nHeight-1-nWinWidth/2,j) += colorWeight.at<double>(i0,j) //s1
																		*distWeight[k];	//*distWeight[i0];
			}
			sum += colorWeight.at<double>(i0,j)
				*distWeight[k]
				*tmp_hor.at<double>(i0,j);
		}
		tmp_ver_d.at<double>(nHeight-1-nWinWidth/2,j) = sum;///(0.001 + total_wght_ver_d.at<double>(nHeight-1-nWinWidth/2,j) );//s2//tmp_ver_d.at<double>(nHeight-1-nWinWidth/2,j) = sum /(0.001+cnt);
		for(int i=nHeight-1;i>=0; i--)//for(int j=nWidth-1;j>=0; j--)
		{
			//注意边界
			if (i+nWinWidth/2+1 >= nHeight || nWinWidth ==1 )continue; //nWinWidth ==1 则跳过不计算

			//tmp_hor_r.at<double>(i,j) = sum;//取平均/(nWinWidth/2)
			//更新窗口内有效像素点数
			total_wght_ver_d.at<double>(i,j) = total_wght_ver_d.at<double>(i+1,j);//s8
			if ( abs(tmp_hor.at<double>(i+nWinWidth/2+1,j)) > 0.001)
			{
				cnt--;
				total_wght_ver_d.at<double>(i,j) -= colorWeight.at<double>(i+nWinWidth/2+1,j) //s3
													*distWeight[nWinWidth-1];
			}
			total_wght_ver_d.at<double>(i,j) *= BETA;//s7
			if ( abs(tmp_hor.at<double>(i+1,j)) > 0.001) 
			{
				cnt++;
				total_wght_ver_d.at<double>(i,j) += colorWeight.at<double>(i+1,j) //s4//////////////////
											*distWeight[nWinWidth/2+1];
			}

			
			sum = (
				sum - colorWeight.at<double>(i+nWinWidth/2+1,j)* distWeight[nWinWidth-1] * tmp_hor.at<double>(i+nWinWidth/2+1,j)
				)*BETA
				+ colorWeight.at<double>(i+1,j)*  distWeight[nWinWidth/2+1] * tmp_hor.at<double>(i+1,j);
			if (cnt == 0) total_wght_ver_d.at<double>(i,j) = 0;//s6 强制约束，防止浮点精度问题
			tmp_ver_d.at<double>(i,j) = sum;/// (0.001+total_wght_ver_d.at<double>(i,j)); //s5//tmp_ver_d.at<double>(i,j) = sum/(0.001 + cnt);

			//static FILE * fout = fopen("tmpVerD.txt","w+");
			//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_ver_d.at<double>(i,j),tmp_ver_d.at<double>(i,j) ,cnt );
			assert( (abs(sum)<0.001&&cnt==0)||(abs(sum)>0.001&&cnt!=0) );
		}

	}


#pragma omp parallel for
	for(int i=0; i<nHeight; i++)
	{
		for(int j=0; j<nWidth; j++)
		{
			double w_total = 0.001 + total_wght_ver_u.at<double>(i,j) + total_wght_ver_d.at<double>(i,j);
			FilterMat.at<double>(i,j) = ( tmp_ver_u.at<double>(i,j)+tmp_ver_d.at<double>(i,j) ) / w_total;//+tmp_ver_d.at<double>(i,j));//0.5*(tmp_ver_d.at<double>(i,j)+tmp_ver_u.at<double>(i,j));// tmp_hor.at<double>(i,j);////tmp_ver_u.at<double>(i,j); //+ tmp_hor_r.at<double>(i,j))*0.5;
		}
	}
}
/////////////////////////////////////////////////////////////////////////////////////
//函数说明:用于深度插值的双边滤波函数
//
/////////////////////////////////////////////////////////////////////////////////////
void BFilterDepthInterp(Mat &pGrayMat, Mat& colorWeight,Mat & FilterMat, int nWidth, int nHeight, double dSigma1, double dSigma2, int nWindows,double INVAILD)  
{  
	////////////////////////参数说明///////////////////////////////////  
	//pGrayMat:待处理图像数组  
	//pFilterMat:保存高斯滤波结果  
	//nWidth:图像宽度  
	//nHeight:图像高度  
	//dSigma1、dSigma2:分别为几何与灰度相关高斯函数的方差  
	double* dDistance = new double[nWindows*nWindows]; //计算距离中间点的几何距离  
	double* dGrayDiff = new double[nWindows*nWindows]; //定义中心点到当前点的灰度差  

	for(int i=0;i<nHeight;i++)
	{
		for(int j=0;j<nWidth;j++)
		{
			double dGray = pGrayMat.at<double>(i,j);
			if (dGray != INVAILD)
				colorWeight.at<double>(i,j) = exp(-0.5*dGray*dGray/dSigma2/dSigma2); //S参数  
			else 
				colorWeight.at<double>(i,j) = 0;
			
		}
	}

	for(int i=0; i<nWindows*nWindows; i++)  
	{  
		int nNumX = i/nWindows;  
		int nNumY = i%nWindows;  
		dDistance[i] = ((nWindows-1)/2-nNumX)*((nWindows-1)/2-nNumX) + ((nWindows-1)/2-nNumY)*((nWindows-1)/2-nNumY);  
		dDistance[i] = exp(-0.5*dDistance[i]/dSigma1/dSigma1);   //C参数  
	}  
//clock_t timer = clock();
	//以下求解灰度值的差  
	for(int i=0; i<nHeight; i++)  
	{  
		for(int j=0; j<nWidth; j++)  
		{  
			double dGray = pGrayMat.at<double>(i,j);//cvmGet(pGrayMat, i, j);    //当前点的灰度值  
			//added
			if (dGray != INVAILD) 
			{
				FilterMat.at<double>(i,j) = dGray;
				continue; //对原始有效深度点位置不进行插值
			}
			//added end
			double dData = 0.0;  
			double dTotal = 0.0;                      //用于进行归一化   
			for(int m=0; m<nWindows*nWindows; m++)  
			{  
				int nNumX = m/nWindows;               //行索引  
				int nNumY = m%nWindows;               //列索引  
				int nX = i-(nWindows-1)/2+nNumX;  
				int nY = j-(nWindows-1)/2+nNumY;  
				if( (nY>=0) && (nY<nWidth) && (nX>=0) && (nX<nHeight))  //边界判断
				{  
					double dGray1 = pGrayMat.at<double>(nX,nY);//cvmGet(pGrayMat, nX, nY);  //x是行序号，y是列序号
					//added
					if (dGray1 != INVAILD  ) //窗口内像素点 深度有效才纳入计算
					{
						
						///dGrayDiff[m] = fabs(dGray-dGray1);  
						///dGrayDiff[m] = exp(-0.5*dGrayDiff[m]*dGrayDiff[m]/dSigma2/dSigma2); //S参数  
						//if (m!= nWindows*nWindows/2)//if(m!=4)  
						{  
							dData += dGray1 * colorWeight.at<double>(nX,nY) * dDistance[m];//dData += dGray1 *  dDistance[m];//dData += dGray1 * colorWeight.at<double>(nX,nY) * dDistance[m];///dData += dGray1 * dGrayDiff[m] * dDistance[m]; //C和S参数综合  
							dTotal +=  colorWeight.at<double>(nX,nY)*dDistance[m];//dTotal += dDistance[m]; //dTotal +=  colorWeight.at<double>(nX,nY)*dDistance[m]; ///dTotal += dGrayDiff[m]*dDistance[m];    //加权系数求和，进行归一化  

						}  
					}
				}  
			}  
			dData /=dTotal;  
			FilterMat.at<double>(i,j) = dData;//cvmSet(pFilterMat, i, j, dData);  
		}  
	}  
//cout<<clock()-timer<<endl;
	delete[]dDistance;    
	delete[]dGrayDiff;  
}  


/////////////////////////////////////////////////////////////////////////////////////
//函数说明:原始的双边滤波函数
//
/////////////////////////////////////////////////////////////////////////////////////

void BilateralFilter(Mat &pGrayMat, Mat & FilterMat, int nWidth, int nHeight, double dSigma1, double dSigma2, int nWindows)  
{  
	////////////////////////参数说明///////////////////////////////////  
	//pGrayMat:待处理图像数组  
	//pFilterMat:保存高斯滤波结果  
	//nWidth:图像宽度  
	//nHeight:图像高度  
	//dSigma1、dSigma2:分别为几何与灰度相关高斯函数的方差  
	double* dDistance = new double[nWindows*nWindows]; //计算距离中间点的几何距离  
	double* dGrayDiff = new double[nWindows*nWindows]; //定义中心点到当前点的灰度差  
	for(int i=0; i<nWindows*nWindows; i++)  
	{  
		int nNumX = i/nWindows;  
		int nNumY = i%nWindows;  
		dDistance[i] = ((nWindows-1)/2-nNumX)*((nWindows-1)/2-nNumX) + ((nWindows-1)/2-nNumY)*((nWindows-1)/2-nNumY);  
		dDistance[i] = exp(-0.5*dDistance[i]/dSigma1/dSigma1);   //C参数  
	}  
	//以下求解灰度值的差  
	for(int i=0; i<nHeight; i++)  
	{  
		for(int j=0; j<nWidth; j++)  
		{  
			double dGray = pGrayMat.at<uchar>(i,j);//cvmGet(pGrayMat, i, j);    //当前点的灰度值  
			double dData = 0.0;  
			double dTotal = 0.0;                      //用于进行归一化   
			for(int m=0; m<nWindows*nWindows; m++)  
			{  
				int nNumX = m/nWindows;               //行索引  
				int nNumY = m%nWindows;               //列索引  
				int nX = i-(nWindows-1)/2+nNumX;  
				int nY = j-(nWindows-1)/2+nNumY;  
				if( (nY>=0) && (nY<nWidth) && (nX>=0) && (nX<nHeight))  
				{  
					double dGray1 = pGrayMat.at<uchar>(nX,nY);//cvmGet(pGrayMat, nX, nY);  
					dGrayDiff[m] = fabs(dGray-dGray1);  
					dGrayDiff[m] = exp(-0.5*dGrayDiff[m]*dGrayDiff[m]/dSigma2/dSigma2); //S参数  
					//if (m!= nWindows*nWindows/2)//if(m!=4)  
					{  
						dData += dGray1 * dGrayDiff[m] * dDistance[m]; //C和S参数综合  
						dTotal += dGrayDiff[m]*dDistance[m];    //加权系数求和，进行归一化  
					}  
				}  
			}  
			dData /=dTotal;  
			FilterMat.at<uchar>(i,j) = dData;//cvmSet(pFilterMat, i, j, dData);  
		}  
	}  
	delete[]dDistance;    
	delete[]dGrayDiff;  
}  
