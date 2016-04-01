#include "FastDepthInterp.h"
using namespace cv;



CFastDepthInterp::CFastDepthInterp(const cv::Mat & pGrayMat_v,const cv::Size & v_imgSize,int v_interpWinSize,double v_sigma1,double v_sigma2)
	:imgSize(v_imgSize),interpWinSize(v_interpWinSize),sigma1(v_sigma1),sigma2(v_sigma2)
{
	pGrayMat = new cv::Mat(imgSize, CV_64FC1,cv::Scalar(0));
	*pGrayMat = pGrayMat_v.clone();
	FilterMat = new cv::Mat(imgSize,CV_64FC1,cv::Scalar(0));
	colorWeight = new cv::Mat(imgSize,CV_64FC1,cv::Scalar(0));

	assert(imgSize == pGrayMat_v.size());

	/*for(int i=0; i<imgSize.height; i++)
		for(int j=0; j<imgSize.width; j++)
		{
			pGrayMat->at<double>(i,j) = pGrayMat_v.at<double>(i,j);
		}*/
	
}
void CFastDepthInterp::fastBFilterDepthInterp( double INVAILD)//(cv::Mat * pGrayMat, cv::Mat * colorWeight,cv::Mat * FilterMat, int nWidth, int nHeight, double sigma1, double sigma2, int nWinWidth,double INVAILD)
{
	int nWidth = imgSize.width;
	int nHeight = imgSize.height;
	int nWinWidth = interpWinSize;
	////////////////////////参数说明///////////////////////////////////  
	//pGrayMat:待处理图像数组  
	//pFilterMat:保存高斯滤波结果  
	//nWidth:图像宽度  
	//nHeight:图像高度  
	//sigma1、sigma2:分别为几何与灰度相关高斯函数的方差  
	const double e = 2.718281828459;
	double * distWeight = new double[nWinWidth];

	const double BETA = exp(-1/sigma1);//const double BETA = exp(-0.5/sigma1);
	const double BETA_INV = exp(1/sigma1);//const double BETA_INV = exp(0.5/sigma1);
	for(int i=0; i<nWinWidth; i++)  
	{  
		distWeight[i] = abs(i-nWinWidth/2);//((nWindows-1)/2-nNumX)*((nWindows-1)/2-nNumX) + ((nWindows-1)/2-nNumY)*((nWindows-1)/2-nNumY);  
		distWeight[i] = exp(-distWeight[i]/sigma1);//exp(-0.5*distWeight[i]/sigma1);   //C参数  

		//static FILE * fout = fopen("tmp3.txt","w+");
		//fprintf(fout,"%f %f \n",distWeight[i] );
	}  

	for(int i=0;i<nHeight;i++)
	{
		for(int j=0;j<nWidth;j++)
		{
			double dGray = pGrayMat->at<double>(i,j);
			//if (dGray != INVAILD)
				colorWeight->at<double>(i,j) = 1;//exp(-0.5*dGray*dGray/sigma2/sigma2); //S参数  
			//else 
				//colorWeight->at<double>(i,j) = 0;

		}
	}
//计算窗口aggregation
	cv::Mat tmp_hor_l(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_hor_r(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_ver_u(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_ver_d(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat tmp_hor(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_hor_l(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_hor_r(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_ver_u(colorWeight->size(),CV_64FC1,Scalar(0));
	cv::Mat total_wght_ver_d(colorWeight->size(),CV_64FC1,Scalar(0));
	//计算 tmp_hor_l
	for(int i=0; i<nHeight; i++)
	{
		double sum = 0;
		int hor_l_cnt = 0;

		for(int j0=0; j0<nWinWidth/2+1; j0++)
		{
			if (pGrayMat->at<double>(i,j0) != 0) //注意！此处认为pGrayMat 值为0 是无效的
			{
				hor_l_cnt++;
				total_wght_hor_l.at<double>(i,nWinWidth/2)+=colorWeight->at<double>(i,j0) //s1
													*distWeight[j0];
			}

			sum += colorWeight->at<double>(i,j0)
					*distWeight[j0]
					*pGrayMat->at<double>(i,j0);

			
		}

		tmp_hor_l.at<double>(i,nWinWidth/2) = sum;///(0.001+total_wght_hor_l.at<double>(i,nWinWidth/2));//s2 //sum / (1e-3+ hor_l_cnt); //sum/(nWinWidth/2+1);//第一个有效赋值
		for(int j=0;j<nWidth; j++)
		{
			//注意边界
			if (j-nWinWidth/2-1 <0)continue;
			//tmp_hor_l.at<double>(i,j) = sum/(nWinWidth/2+1);
			
			//更新窗口内有效像素点数
			total_wght_hor_l.at<double>(i,j) = total_wght_hor_l.at<double>(i,j-1);//s8 赋初值
			if (pGrayMat->at<double>(i,j-nWinWidth/2-1) != 0)
			{
					hor_l_cnt--;
					total_wght_hor_l.at<double>(i,j) -= colorWeight->at<double>(i,j-nWinWidth/2-1) //s3
														*distWeight[0];
			}
			total_wght_hor_l.at<double>(i,j) *= BETA;//s7
			if (pGrayMat->at<double>(i,j) != 0) 
			{
					hor_l_cnt++;
					total_wght_hor_l.at<double>(i,j) += colorWeight->at<double>(i,j) //s4
															*distWeight[nWinWidth/2];
			}
			
			sum = (
					sum - colorWeight->at<double>(i,j-nWinWidth/2-1)* distWeight[0] * pGrayMat->at<double>(i,j-nWinWidth/2-1)
				   )*BETA
				   + colorWeight->at<double>(i,j)*  distWeight[nWinWidth/2] * pGrayMat->at<double>(i,j);

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
	//		if (pGrayMat->at<uchar>(i,j0) != 0) hor_r_cnt++;

	//		sum += colorWeight->at<double>(i,j0)
	//			*distWeight[j0+nWinWidth/2+1]
	//			*pGrayMat->at<uchar>(i,j0);
	//	}
	//	tmp_hor_r.at<double>(i,0) = sum/(0.001+hor_r_cnt);
	//	for(int j=1;j<nWidth; j++)
	//	{
	//		//注意边界
	//		if (j+nWinWidth/2 >= nWidth)continue;
	//		//tmp_hor_r.at<double>(i,j) = sum;//取平均/(nWinWidth/2)
	//		//更新窗口内有效像素点数
	//		if (pGrayMat->at<uchar>(i,j) != 0) hor_r_cnt--;
	//		if (pGrayMat->at<uchar>(i,j+nWinWidth/2) != 0) hor_r_cnt++;

	//		static FILE * fout = fopen("tmp.txt","w+");
	//		fprintf(fout,"%f %f \n",(sum - colorWeight->at<double>(i,j)* distWeight[nWinWidth/2+1] * pGrayMat->at<uchar>(i,j))*BETA_INV,sum );
	//		sum = (
	//			sum - colorWeight->at<double>(i,j)* distWeight[nWinWidth/2+1] * pGrayMat->at<uchar>(i,j)
	//			)*BETA_INV
	//			+ colorWeight->at<double>(i,j+nWinWidth/2)*  distWeight[nWinWidth-1] * pGrayMat->at<uchar>(i,j+nWinWidth/2);
	//		tmp_hor_r.at<double>(i,j) = sum/(0.001+hor_r_cnt);
	//	}

	//}

	///计算 tmp_hor_r //反向计算
	for(int i=0; i<nHeight; i++)
	{
		double sum = 0;
		int hor_r_cnt = 0;
		for(int j0=nWidth-1, k=nWinWidth-1;j0 > nWidth-1-nWinWidth/2;j0--,k--)//for(int j0=nWinWidth/2+1; j0<nWinWidth; j0++)
		{
			if (pGrayMat->at<double>(i,j0) != 0)
			{
					hor_r_cnt++;
					total_wght_hor_r.at<double>(i,nWidth-1-nWinWidth/2)+=colorWeight->at<double>(i,j0) //s1
																		*distWeight[k];//*distWeight[j0];
			}

			sum += colorWeight->at<double>(i,j0)
				*distWeight[k]
				*pGrayMat->at<double>(i,j0);
		}
		tmp_hor_r.at<double>(i,nWidth-1-nWinWidth/2) = sum;///(0.001 + total_wght_hor_r.at<double>(i,nWidth-1-nWinWidth/2));//s2 //tmp_hor_r.at<double>(i,nWidth-1-nWinWidth/2) = sum /(0.001+hor_r_cnt);
		for(int j=nWidth-1;j>=0; j--)
		{
			//注意边界
			if (j+nWinWidth/2+1 >= nWidth || nWinWidth ==1 )continue; //nWinWidth ==1 则跳过不计算

			//tmp_hor_r.at<double>(i,j) = sum;//取平均/(nWinWidth/2)
			//更新窗口内有效像素点数
			total_wght_hor_r.at<double>(i,j) = total_wght_hor_r.at<double>(i,j+1);//s8

			if (pGrayMat->at<double>(i,j+nWinWidth/2+1) != 0) 
			{
				hor_r_cnt--;
				total_wght_hor_r.at<double>(i,j) -= colorWeight->at<double>(i,j+nWinWidth/2+1) //s3
													*distWeight[nWinWidth-1];
			}
			total_wght_hor_r.at<double>(i,j) *= BETA;//s7
			if ( pGrayMat->at<double>(i,j+1) != 0)
			{
				hor_r_cnt++;
				total_wght_hor_r.at<double>(i,j) += colorWeight->at<double>(i,j+1) //s4
					*distWeight[nWinWidth/2+1];
			}

			sum = (
				sum - colorWeight->at<double>(i,j+nWinWidth/2+1)* distWeight[nWinWidth-1] * pGrayMat->at<double>(i,j+nWinWidth/2+1)
				)*BETA
				+ colorWeight->at<double>(i,j+1)*  distWeight[nWinWidth/2+1] * pGrayMat->at<double>(i,j+1);

			if (hor_r_cnt == 0) total_wght_hor_r.at<double>(i,j) = 0;//s6 强制约束，防止浮点精度问题
			tmp_hor_r.at<double>(i,j) = sum;/// (0.001+total_wght_hor_r.at<double>(i,j)); //s5 //tmp_hor_r.at<double>(i,j) = sum/(0.001+hor_r_cnt);
			//static FILE * fout = fopen("tmpHorR.txt","w+");
			//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_hor_r.at<double>(i,j),tmp_hor_r.at<double>(i,j) ,hor_r_cnt);
			assert( (abs(sum)<0.001&&hor_r_cnt==0)||(abs(sum)>0.001&&hor_r_cnt!=0) );
		}

	}
	//...


	//计算融合以上2个矩阵的 FilterMat
	for(int i=0; i<nHeight; i++)
	{
		for(int j=0; j<nWidth; j++)
		{
			//tmp_hor_l.at<double>(i,j) /= ( 0.001+total_wght_hor_l.at<double>(i,j) );
			double w_total = ( 0.001+total_wght_hor_l.at<double>(i,j) + total_wght_hor_r.at<double>(i,j) );
			tmp_hor.at<double>(i,j) =  (tmp_hor_l.at<double>(i,j) + tmp_hor_r.at<double>(i,j) ) / w_total;//(tmp_hor_l.at<double>(i,j) + tmp_hor_r.at<double>(i,j))*0.5;//FilterMat->at<uchar>(i,j) =  (tmp_hor_l.at<double>(i,j) + tmp_hor_r.at<double>(i,j))*0.5;
		}
	}

	// 纵向计算
	//计算 tmp_ver_u
	for(int j=0; j<nWidth; j++)//for(int i=0; i<nHeight; i++)
	{
		double sum = 0;
		int ver_u_cnt = 0;

		for(int i0=0; i0<nWinWidth/2+1; i0++)//for(int j0=0; j0<nWinWidth/2+1; j0++)
		{
			if (abs( tmp_hor.at<double>(i0,j) )>0.001 ) 
			{
				ver_u_cnt++;
				total_wght_ver_u.at<double>(nWinWidth/2,j)+=colorWeight->at<double>(i0,j) //s1
					*distWeight[i0];
			}
			sum += colorWeight->at<double>(i0,j)
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
				total_wght_ver_u.at<double>(i,j) -= colorWeight->at<double>(i-nWinWidth/2-1,j) //s3
					*distWeight[0];
			}
			total_wght_ver_u.at<double>(i,j) *= BETA;//s7
			if (abs( tmp_hor.at<double>(i,j) ) >0.001) 
			{
				ver_u_cnt++;
				total_wght_ver_u.at<double>(i,j) += colorWeight->at<double>(i,j) //s4
					*distWeight[nWinWidth/2];
			}
			sum = (
				sum - colorWeight->at<double>(i-nWinWidth/2-1,j)* distWeight[0] * tmp_hor.at<double>(i-nWinWidth/2-1,j)
				)*BETA
				+ colorWeight->at<double>(i,j)*  distWeight[nWinWidth/2] * tmp_hor.at<double>(i,j);

			if (ver_u_cnt == 0) total_wght_ver_u.at<double>(i,j) = 0;//s6 强制约束，防止浮点精度问题
			tmp_ver_u.at<double>(i,j) = sum;/// (0.001+total_wght_ver_u.at<double>(i,j)); //s5 //tmp_ver_u.at<double>(i,j) = sum/ (0.001+ver_u_cnt);//sum/(1e-6+ hor_l_cnt);///sum/(nWinWidth/2+1);
			//cout<<i<<' '<<j<<' '<<sum<<' '<<ver_u_cnt<<endl;
			//static FILE * fout = fopen("tmp2.txt","w+");///////////////////////////
			//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_ver_u.at<double>(i,j),tmp_ver_u.at<double>(i,j) ,ver_u_cnt );////////////////
			assert( (abs(sum)<0.001&&ver_u_cnt==0)||(abs(sum)>0.001&&ver_u_cnt!=0) );
		}

	}
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
				total_wght_ver_d.at<double>(nHeight-1-nWinWidth/2,j) += colorWeight->at<double>(i0,j) //s1
																		*distWeight[k];	//*distWeight[i0];
			}
			sum += colorWeight->at<double>(i0,j)
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
				total_wght_ver_d.at<double>(i,j) -= colorWeight->at<double>(i+nWinWidth/2+1,j) //s3
													*distWeight[nWinWidth-1];
			}
			total_wght_ver_d.at<double>(i,j) *= BETA;//s7
			if ( abs(tmp_hor.at<double>(i+1,j)) > 0.001) 
			{
				cnt++;
				total_wght_ver_d.at<double>(i,j) += colorWeight->at<double>(i+1,j) //s4//////////////////
											*distWeight[nWinWidth/2+1];
			}

			
			sum = (
				sum - colorWeight->at<double>(i+nWinWidth/2+1,j)* distWeight[nWinWidth-1] * tmp_hor.at<double>(i+nWinWidth/2+1,j)
				)*BETA
				+ colorWeight->at<double>(i+1,j)*  distWeight[nWinWidth/2+1] * tmp_hor.at<double>(i+1,j);
			if (cnt == 0) total_wght_ver_d.at<double>(i,j) = 0;//s6 强制约束，防止浮点精度问题
			tmp_ver_d.at<double>(i,j) = sum;/// (0.001+total_wght_ver_d.at<double>(i,j)); //s5//tmp_ver_d.at<double>(i,j) = sum/(0.001 + cnt);

			//static FILE * fout = fopen("tmpVerD.txt","w+");
			//fprintf(fout,"%lf %lf %lf %d\n",sum, total_wght_ver_d.at<double>(i,j),tmp_ver_d.at<double>(i,j) ,cnt );
			assert( (abs(sum)<0.001&&cnt==0)||(abs(sum)>0.001&&cnt!=0) );
		}

	}




	for(int i=0; i<nHeight; i++)
	{
		for(int j=0; j<nWidth; j++)
		{
			double w_total = 0.001 + total_wght_ver_u.at<double>(i,j) + total_wght_ver_d.at<double>(i,j);
			FilterMat->at<double>(i,j) = ( tmp_ver_u.at<double>(i,j)+tmp_ver_d.at<double>(i,j) ) / w_total;//+tmp_ver_d.at<double>(i,j));//0.5*(tmp_ver_d.at<double>(i,j)+tmp_ver_u.at<double>(i,j));// tmp_hor.at<double>(i,j);////tmp_ver_u.at<double>(i,j); //+ tmp_hor_r.at<double>(i,j))*0.5;
		}
	}
}