#include "Nonlocal.h"
#include"stack"
//#define CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>
Nonlocal::Nonlocal()
{
	//m_buf_u2=NULL;
	m_buf_d3=NULL;
	m_left=NULL;
}
Nonlocal::~Nonlocal()
{
    clean();
}
void Nonlocal::clean()
{
//	freeu_3(m_buf_u2); m_buf_u2=NULL;
	//freeu_4(m_buf_u3); m_buf_u3=NULL;
	//freef_3(m_buf_f2); m_buf_f2=NULL;
	freef_4(m_buf_d3); m_buf_d3=NULL;
	freeu_3(m_left);m_left=NULL;
}
int Nonlocal::init(int h,int w,int nr_plane,double sigma_range)
{
	clean();
	m_h=h; m_w=w; m_nr_plane=nr_plane; m_sigma_range=sigma_range;
	m_buf_d3=allocf_4(2,m_h,m_w,m_nr_plane);
	m_cost_vol=m_buf_d3[0];
	m_cost_vol_temp=m_buf_d3[1];
//	m_buf_u2=allocu_3(1,m_h,m_w);
//	m_disparity=m_buf_u2[0];
	m_left=allocu_3(m_h,m_w,3);
	for(int i=0;i<256;i++) m_table[i]=exp(-double(i)/(m_sigma_range*255));

    return(0);
}

void Nonlocal::CostAggregation(bool refinement)
{
//	if(!refinement)
//	{
	m_tf.build_tree(m_left[0][0]);
	m_tf.filter(m_cost_vol[0][0],m_cost_vol_temp[0][0],m_nr_plane);

//	depth_best_cost(m_disparity,m_cost_vol,m_h,m_w,m_nr_plane);

	//ctmf(m_disparity[0],disparity[0],m_w,m_h,m_w,m_w,radius,1,m_h*m_w);
	//image_copy(m_disparity,disparity,m_h,m_w);
// 	if(0)
// 	{
// 		depth_best_cost(m_disparity,m_cost_vol,m_h,m_w,m_nr_plane);
// 	//	image_copy(m_cost_vol,m_cost_vol_right,m_h,m_w,m_nr_plane);
// 	//	m_tf_right.filter(m_cost_vol[0][0],m_cost_vol_temp[0][0],m_nr_plane);
// 	//	depth_best_cost(m_disparity,m_cost_vol,m_h,m_w,m_nr_plane);
// 	//	ctmf(m_disparity[0],m_disparity_right[0],m_w,m_h,m_w,m_w,radius,1,m_h*m_w);
// 	//image_display(m_disparity_right,m_h,m_w);
// 	//image_display(disparity,m_h,m_w);
// 	//occlusion_solver_left_right(disparity,m_disparity_right,m_h,m_w,m_nr_plane,false);
// 	//detect_occlusion_left_right(m_mask_occlusion,disparity,m_disparity_right,m_h,m_w,m_nr_plane);
//     m_detectUnstable(block);
// 	image_zero(m_cost_vol,m_h,m_w,m_nr_plane);
// 		//int th=int(0.1*m_nr_plane+0.5);
// 		for(int y=0;y<m_h;y++)
// 			for(int x=0;x<m_w;x++)
// 			{
// 			//	printf("mask=%d\n",m_mask_occlusion[y][x]);
// 				if(m_mask_occlusion[y][x]!=255)
// 				{
// 					for(int d=0;d<m_nr_plane;d++)
// 						m_cost_vol[y][x][d]=abs(m_disparity[y][x]-d);
// 				}
// 			}
// 		//	depth_best_cost(::)
// 		m_tf.update_table(m_sigma_range);
// 		m_tf.filter(m_cost_vol[0][0],m_cost_vol_temp[0][0],m_nr_plane);
// 		depth_best_cost(m_disparity_right,m_cost_vol,m_h,m_w,m_nr_plane);
// 		ctmf(m_disparity_right[0],m_disparity[0],m_w,m_h,m_w,m_w,radius,1,m_h*m_w);
// 	}
 //   ;
}

void Nonlocal::stereo(CMDRVideoFrame* pCurrentFrame,double sigma,CBlock block,CDataCost & datacost,ZIntImage& labelimg,bool rfm)
{
	//	if(!rfm)
	//	{
//	ZIntImage zimg;
//	zimg.CreateAndInit(block.m_iWidth,block.m_iHeight,3,0);
	init(block.m_iHeight,block.m_iWidth,datacost.GetChannel(),sigma);
	m_tf.init(m_h,m_w,3,m_sigma_range,4);
	for(int idx=0;idx<block.m_iWidth;idx++)
		for(int idy=0;idy<block.m_iHeight;idy++)
		{
			Wml::Vector3d outColor;
			pCurrentFrame->GetColorAt(block.m_X+idx,block.m_Y+idy,outColor);

			union uname u1;
			u1.limit_data =  ((int)(outColor[0]));
			m_left[idy][idx][0]=u1.buffer[0];
	//		zimg.at(idx,idy,0)=m_left[idy][idx][0];

			u1.limit_data =  ((int)(outColor[1]));
			m_left[idy][idx][1]=u1.buffer[0];
		//	zimg.at(idx,idy,1)=m_left[idy][idx][1];

			u1.limit_data =  ((int)(outColor[2]));
			m_left[idy][idx][2]=u1.buffer[0];
	//		zimg.at(idx,idy,2)=m_left[idy][idx][2];

	//		cout<<(int)m_left[idy][idx][0]<<"  "<<(int)(int)m_left[idy][idx][1]<<"  "<<m_left[idy][idx][2]<<endl;
			for(int idsp=0;idsp<datacost.GetChannel();idsp++)
			{
				m_cost_vol[idy][idx][idsp]=datacost.At(idx,idy,idsp);
	//	if(datacost.At(idx-block.m_X,idy-block.m_Y,idsp) <= 40 && datacost.At(idx-block.m_X,idy-block.m_Y,idsp) >= 0);
	//	else cout<<idx<<"  "<<idy<<"  "<<idsp<<"  "<<datacost.At(idx-block.m_X,idy-block.m_Y,idsp)<<endl;
			}
		}
	//	SaveZimg(zimg,"zimg");
		//	}
		// 	else
		// 		for(int idx=0;idx<block.m_iWidth;idx++)
		// 			for(int idy=0;idy<block.m_iHeight;idy++)
		// 		for(int idsp=0;idsp<datacost.GetChannel();idsp++)
		// 		{
		// 			m_cost_vol[idy][idx][idsp]=datacost.At(idx,idy,idsp);
		// 		}
		CostAggregation(rfm);
		for(int idx=block.m_X;idx<block.m_X+block.m_iWidth;idx++)
			for(int idy=block.m_Y;idy<block.m_Y+block.m_iHeight;idy++)
			{
				labelimg.at(idx,idy)=0;
				double min_val=1e23;
				for(int idsp=0;idsp<datacost.GetChannel();idsp++)
				{
					datacost.At(idx-block.m_X,idy-block.m_Y,idsp)=m_cost_vol[idy-block.m_Y][idx-block.m_X][idsp];
					//cout<<datacost.At(idx-block.m_X,idy-block.m_Y,idsp)<<endl;
					if( datacost.At(idx-block.m_X,idy-block.m_Y,idsp) < min_val )
					{
						min_val=datacost.At(idx-block.m_X,idy-block.m_Y,idsp);
					    labelimg.at(idx,idy)=idsp;
					}
				}
			}
			clean();
			//char* test=new char[20];
}

// void Nonlocal::m_detectUnstable(CBlock& block)
// {
// //	image_zero(m_cost_vol,m_h,m_w,m_nr_plane);
// 	int frameCount = m_lf.m_iFramCount;
// 	int dspLevel = DepthPara::GetInstance()->m_iDspLevel;
// 	double dspMin=DepthPara::GetInstance()->m_dDspMin;
// 	double dspMax=DepthPara::GetInstance()->m_dDspMax;
// 	std::vector<double> dspV(dspLevel);
// 	for(auto i=0;i<dspLevel;i++)
// 		dspV[i] = dspMin * (dspLevel - 1 - i)/(dspLevel - 1)  + dspMax * i/(dspLevel - 1) ;
// 	memset(m_mask_occlusion[0],0,sizeof(char)*block.m_iHeight*block.m_iWidth);
// 	//CxImage dspimg;
// 	//dspimg.Create(block.m_iWidth,block.m_iHeight,32);
// 	IplImage *dspimg=cvCreateImage(cvSize(block.m_iWidth,block.m_iHeight),IPL_DEPTH_8U,1);
//
// 	ZIntImage dsparity;
// 	m_pCurrentFrame->InitLabelImgByDspImg(dsparity,dspLevel,dspMin,dspMax, block);
//
// 	for(int h=0;h<block.m_iHeight;h++)
// 	{
// 		printf("%d ",h);
// 		for(int w=0;w<block.m_iWidth;w++)
// 		{
// 			int bestLabel=0;
// 			double MaxLikelihood = 1e-20F;
// 			Wml::Vector3d ptWorldCoord;
// 			Wml::Vector3d CurrentColor, CorrespondingColor;
// 			//m_cost_vol[y][x][i]
// 			int realFrameCount=0;
// 			m_pCurrentFrame->GetColorAt(block.m_X+w, block.m_Y+h, CurrentColor);
//
// 			   int depthLeveli=dsparity.at(w,h);
// 			   double dspvv=m_pCurrentFrame->GetDspAt(block.m_X+w,block.m_Y+h);
// 			 //
// 			//	   printf("%f",dspvv);
// 				realFrameCount=0;
// 				int leftRealFrameCount=0,rightRealFrameCount=0;
// 				double leftwc=0,rightwc=0;
// 				m_pCurrentFrame->GetWorldCoordFrmImgCoord(block.m_X+w, block.m_Y+h, dspvv, ptWorldCoord);
// 				//double z = 1.0/dspV[depthLeveli];
// 	//		 if(h==523&& w==439)
// 		//		 printf("\n %d    %f",w,dspvv);
// 				for(int i=m_pCurrentFrame->m_iID-1; i<m_pCurrentFrame->m_iID+2 ; i++)
// 				{
// 					if(i<0 || i>=m_lf.GetFrameCount()) continue;
// 					if(i==m_pCurrentFrame->m_iID) continue;
// 					double dsp = 0, x2, y2;
// 					m_lf.m_frames[i]->GetImgCoordFrmWorldCoord(x2, y2, dsp, ptWorldCoord);
// 					//	m_pNearFrames->at(i)->GetImgCoordFrmWorldCoord(x2, y2, dsp, ptWorldCoord);
// 					//z2 = r[i][2]*z + b[i][2];
// 					//u2 = (r[i][0]*z + b[i][0]) / z2;
// 					//v2 = (r[i][1]*z + b[i][1]) / z2;
//
// 					//	dataCosti[depthLeveli] += m_dColorSigma / (m_dColorSigma + (m_dColorMissPenalty/3)*(m_dColorMissPenalty/3));
// 					if(x2>0 && y2>0 && x2 < CMDRVideoFrame::GetImgWidth()-1 && y2 < CMDRVideoFrame::GetImgHeight()-1)
// 					{
// 						m_lf.m_frames[i]->GetColorAt(x2, y2, CorrespondingColor);
// 						double colordist = (fabs(CurrentColor[0] - CorrespondingColor[0])
// 							+ fabs(CurrentColor[1] - CorrespondingColor[1])
// 							+ fabs(CurrentColor[2] - CorrespondingColor[2])) / 3.0;
//
// 						colordist = min(30.0, colordist);
// 						//double wc = m_dColorSigma/(m_dColorSigma + colordist);
//
// 						double wc =25.0/(25.0 + colordist*colordist);
//
// 						double d2 = m_lf.m_frames[i]->GetDspAt(x2, y2);
// 						//double d2_INT = m_lf.m_frames[i]->GetDspAt((int)(x2+0.5F), (min((int)(y2+0.5F), 559)));
// 						double d2_INT = m_lf.m_frames[i]->GetDspAt((int)(x2+0.5F), (int)(y2+0.5F));
// 						//dsp = 1.0/z2;
// 						double dspDiff = min(fabs(d2 - dsp), fabs(d2_INT - dsp));
// 						double dspSigma = (dspMax-dspMin) * 2.0/100.0;
// 						double dspSigma2=dspSigma*dspSigma;
// 						double wd = dspSigma2 / (dspSigma2 + dspDiff*dspDiff);
// 						wc=wc*wd;
// 						wc=max(1e-10, wc );
//
// 						if(i<m_pCurrentFrame->m_iID) {leftwc+=wc; leftRealFrameCount++;}
// 						else{ rightwc+=wc; rightRealFrameCount++;}
// 					}
// 					else ;
// 				}
// 				double trust=0;
//
// 				if(rightRealFrameCount==0||leftRealFrameCount==0);
// 				else
// 				  trust=max(leftwc/leftRealFrameCount,rightwc/rightRealFrameCount);
// //				printf("trust=%f\n",trust);
// 				if(trust<0.7)
// 			        	m_mask_occlusion[h][w]=255;
// 			CvScalar s;
// 			if(m_mask_occlusion[h][w]!=255)
// 			{
// 				s.val[0]=depthLeveli*255.0/dspLevel;
// 				s.val[1]=0;
// 				s.val[2]=0;
// 			}
// 			else
// 			{
// 				s.val[0]=255;
// 				s.val[1]=0;
// 				s.val[2]=0;
// 			}
// 			cvSet2D(dspimg,h,w,s);
// 		}
// 	}
//
// // 	cvNamedWindow("dsp",1);
// // 	cvShowImage("dsp",dspimg);
// // 	cvWaitKey(0);
// }
//