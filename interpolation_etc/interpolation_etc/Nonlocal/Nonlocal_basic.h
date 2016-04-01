#ifndef CVPR09_CTBF_BASIC_H
#define CVPR09_CTBF_BASIC_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numeric>
#include <vector>
#include <process.h>
#include <direct.h>
#include <io.h>
#include <time.h>
#include <string>
#include <memory.h>
#include <algorithm>
#include <functional>      // For greater<int>()
#include <iostream>
//#include"ZImageUtil.h"
//#include"MDRVideoFrame.h"
#if _MSC_VER > 1020   // if VC++ version is > 4.2
   using namespace std;  // std c++ libs implemented in std
#endif
#define DEF_PADDING					10
#define DEF_THRESHOLD_ZERO			1e-6
#define DEF_PI_DOUBLE				3.14159265359
#define DEF_FLOAT_MAX				1.175494351e+38F
#define DEF_DOUBLE_MAX				1.7E+308
#define DEF_FLOAT_RELATIVE_ACCURACY	2.2204e-016
#define DEF_INI_MAX					2147483647
#define DEF_SHORT_MAX				65535
#define DEF_CHAR_MAX					255
#define	DEF_SEED						42
#define DEF_THRESHOLD_ZERO			1e-6
#define DEF_THRESHOLD_ZERO_DOUBLE	1e-16
#define DEF_ENTER					10
#define DEF_BLANK					32
#define DEF_STRING_LENGTH			300

   enum RUNTYPE{
	   RUN_INIT,
	   RUN_BO,
	   RUN_BOPF,
	   RUN_NOLOCAL,
	   RUN_DE,
	   RUN_RefineByModel,
	   RUN_ExportModelPoisson,
	   RUN_TextureMapping,
	   RUN_ExportModelAndActs
   };
class timer{public: void start(); double stop(); void time_display(char *disp="",int nr_frame=1); void fps_display(char *disp="",int nr_frame=1); private: double m_pc_frequency; __int64 m_counter_start;};//clock_t m_begin,m_end;};

inline float max_f3(float*a){return(max(max(a[0],a[1]),a[2]));}
inline float min_f3(float*a){return(min(min(a[0],a[1]),a[2]));}
inline double div(double x,double y){return((y!=0)?(x/y):0);}
/*Box filter*/
void boxcar_sliding_window_x(double *out,double *in,int h,int w,int radius);
void boxcar_sliding_window_y(double *out,double *in,int h,int w,int radius);
void boxcar_sliding_window(double **out,double **in,double **temp,int h,int w,int radius);
void boxcar_sliding_window(float**out,float**in,float**temp,int h,int w,int radius);
void boxcar_sliding_window(unsigned char**out,unsigned char**in,unsigned char**temp,int h,int w,int radius);
/*Gaussian filter*/
int gaussian_recursive(double **image,double **temp,double sigma,int order,int h,int w);
void gaussian_recursive_x(double **od,double **id,int w,int h,double a0,double a1,double a2,double a3,double b1,double b2,double coefp,double coefn);
void gaussian_recursive_y(double **od,double **id,int w,int h,double a0,double a1,double a2,double a3,double b1,double b2,double coefp,double coefn);
int gaussian_recursive(float **image,float **temp,float sigma,int order,int h,int w);
/*basic functions*/

inline void image_dot_product(double*out,float*a,float*b,int len){for(int i=0;i<len;i++)*out++=double(*a++)*double(*b++);}
inline void image_dot_product(double*out,float*a,unsigned char*b,int len){for(int i=0;i<len;i++)*out++=double(*a++)*double(*b++);}
inline void image_dot_product(double*out,double*a,double*b,int len){for(int i=0;i<len;i++)*out++=(*a++)*(*b++);}
//inline float min(float a,float b){if(a<b) return(a); else return(b);}
//inline float max(float a,float b){if(a>b) return(a); else return(b);}
inline int sum_u3(unsigned char *a) {return(a[0]+a[1]+a[2]);}
inline double sum_d3(double*a){return(a[0]+a[1]+a[2]);}
inline unsigned char min_u3(unsigned char *a){return(min(min(a[0],a[1]),a[2]));}
inline unsigned char max_u3(unsigned char *a){return(max(max(a[0],a[1]),a[2]));}
inline unsigned char max_u3(unsigned char r,unsigned char g,unsigned char b){return(max(max(r,g),b));}
inline void image_zero(float **in,int h,int w,float zero=0){memset(in[0],zero,sizeof(float)*h*w);}
inline void image_zero(double **in,int h,int w,double zero=0){memset(in[0],zero,sizeof(double)*h*w);}
inline void image_zero(unsigned char**in,int h,int w,unsigned char zero=0){memset(in[0],zero,sizeof(unsigned char)*h*w);}
inline void image_zero(double ***in,int h,int w,int d,double zero=0){memset(in[0][0],zero,sizeof(double)*h*w*d);}
inline unsigned char rgb_2_gray(unsigned char*in){return(unsigned char(0.299*in[0]+0.587*in[1]+0.114*in[2]+0.5));}
inline int square_difference_u3(unsigned char *a,unsigned char *b){int d1,d2,d3; d1=(*a++)-(*b++); d2=(*a++)-(*b++);	d3=(*a++)-(*b++); return(int(d1*d1+d2*d2+d3*d3));}
void specular_free_image(unsigned char ***image_specular_free,unsigned char ***image_normalized,float **diffuse_chromaticity_max,int h,int w);

inline void sort_increase_using_histogram(int*id,unsigned char*image,int len)
{
	int histogram[DEF_CHAR_MAX+1];
	int nr_bin=DEF_CHAR_MAX+1;
	memset(histogram,0,sizeof(int)*nr_bin);
	for(int i=0;i<len;i++)
	{
		histogram[image[i]]++;
	}
	int nr_hitted_prev=histogram[0];
	histogram[0]=0;
	for(int k=1;k<nr_bin;k++)
	{
		int nr_hitted=histogram[k];
		histogram[k]=nr_hitted_prev+histogram[k-1];
		nr_hitted_prev=nr_hitted;
	}
	for(int i=0;i<len;i++)
	{
		unsigned char dist=image[i];
		int index=histogram[dist]++;
		id[index]=i;
	}
}
inline double *get_color_weighted_table(double sigma_range,int len)
{
	double *table_color,*color_table_x; int y;
	table_color=new double [len];
	color_table_x=&table_color[0];
	for(y=0;y<len;y++) (*color_table_x++)=exp(-double(y*y)/(2*sigma_range*sigma_range));
	return(table_color);
}
inline void color_weighted_table_update(double *table_color,double dist_color,int len)
{
	double *color_table_x; int y;
	color_table_x=&table_color[0];
	for(y=0;y<len;y++) (*color_table_x++)=exp(-double(y*y)/(2*dist_color*dist_color));
}

inline void vec_min_val(float &min_val,float *in,int len)
{
	min_val=in[0];
	for(int i=1;i<len;i++) if(in[i]<min_val) min_val=in[i];
}
inline void vec_min_val(unsigned char &min_val,unsigned char *in,int len)
{
	min_val=in[0];
	for(int i=1;i<len;i++) if(in[i]<min_val) min_val=in[i];
}
inline void vec_max_val(float &max_val,float *in,int len)
{
	max_val=in[0];
	for(int i=1;i<len;i++) if(in[i]>max_val) max_val=in[i];
}
inline void vec_max_val(unsigned char &max_val,unsigned char *in,int len)
{
	max_val=in[0];
	for(int i=1;i<len;i++) if(in[i]>max_val) max_val=in[i];
}
inline void down_sample_1(unsigned char **out,unsigned char **in,int h,int w,int scale_exp)
{
	int y,x; int ho,wo; unsigned char *out_y,*in_x;
	ho=(h>>scale_exp); wo=(w>>scale_exp);
	for(y=0;y<ho;y++)
	{
		out_y=&out[y][0]; in_x=in[y<<scale_exp];
		for(x=0;x<wo;x++) *out_y++=in_x[x<<scale_exp];
	}
}
inline void down_sample_1(float**out,float**in,int h,int w,int scale_exp)
{
	int y,x; int ho,wo; float *out_y,*in_x;
	ho=(h>>scale_exp); wo=(w>>scale_exp);
	for(y=0;y<ho;y++)
	{
		out_y=&out[y][0]; in_x=in[y<<scale_exp];
		for(x=0;x<wo;x++) *out_y++=in_x[x<<scale_exp];
	}
}
inline double linear_interpolate_xy(double **image,double x,double y,int h,int w)
{
	int x0,xt,y0,yt; double dx,dy,dx1,dy1,d00,d0t,dt0,dtt;
	x0=int(x); xt=min(x0+1,w-1); y0=int(y); yt=min(y0+1,h-1);
	dx=x-x0; dy=y-y0; dx1=1-dx; dy1=1-dy; d00=dx1*dy1; d0t=dx*dy1; dt0=dx1*dy; dtt=dx*dy;
	return(d00*image[y0][x0]+d0t*image[y0][xt]+dt0*image[yt][x0]+dtt*image[yt][xt]);
}
/*memory*/
inline double *** allocd_3(int n,int r,int c,int padding=DEF_PADDING)
{
	double *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(double*) malloc(sizeof(double)*(n*rc+padding));
	if(a==NULL) {printf("allocd_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(double**) malloc(sizeof(double*)*n*r);
    pp=(double***) malloc(sizeof(double**)*n);
    for(i=0;i<n;i++)
        for(j=0;j<r;j++)
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++)
        pp[i]=&p[i*r];
    return(pp);
}
inline void freed_3(double ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char** allocu(int r,int c,int padding=DEF_PADDING)
{
	unsigned char *a,**p;
	a=(unsigned char*) malloc(sizeof(unsigned char)*(r*c+padding));
	if(a==NULL) {printf("allocu() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(unsigned char**) malloc(sizeof(unsigned char*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void freeu(unsigned char **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char *** allocu_3(int n,int r,int c,int padding=DEF_PADDING)
{
	unsigned char *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(unsigned char*) malloc(sizeof(unsigned char )*(n*rc+padding));
	if(a==NULL) {printf("allocu_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(unsigned char**) malloc(sizeof(unsigned char*)*n*r);
    pp=(unsigned char***) malloc(sizeof(unsigned char**)*n);
    for(i=0;i<n;i++)
        for(j=0;j<r;j++)
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++)
        pp[i]=&p[i*r];
    return(pp);
}
inline void freeu_3(unsigned char ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline void freeu_1(unsigned char*p)
{
	if(p!=NULL)
	{
		delete [] p;
		p=NULL;
	}
}
inline float** allocf(int r,int c,int padding=DEF_PADDING)
{
	float *a,**p;
	a=(float*) malloc(sizeof(float)*(r*c+padding));
	if(a==NULL) {printf("allocf() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(float**) malloc(sizeof(float*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void freef(float **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline float *** allocf_3(int n,int r,int c,int padding=DEF_PADDING)
{
	float *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(float*) malloc(sizeof(float)*(n*rc+padding));
	if(a==NULL) {printf("allocf_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(float**) malloc(sizeof(float*)*n*r);
    pp=(float***) malloc(sizeof(float**)*n);
    for(i=0;i<n;i++)
        for(j=0;j<r;j++)
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++)
        pp[i]=&p[i*r];
    return(pp);
}
inline void freef_3(float ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline int** alloci(int r,int c,int padding=DEF_PADDING)
{
	int *a,**p;
	a=(int*) malloc(sizeof(int)*(r*c+padding));
	if(a==NULL) {printf("alloci() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(int**) malloc(sizeof(int*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void freei(int **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline void freei_1(int*p)
{
	if(p!=NULL)
	{
		delete [] p;
		p=NULL;
	}
}
inline double** allocd(int r,int c,int padding=DEF_PADDING)
{
	double *a,**p;
	a=(double*) malloc(sizeof(double)*(r*c+padding));
	if(a==NULL) {printf("allocd() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(double**) malloc(sizeof(double*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void freed(double **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char**** allocu_4(int t,int n,int r,int c,int padding=DEF_PADDING)
{
	unsigned char *a,**p,***pp,****ppp;
    int nrc=n*r*c,nr=n*r,rc=r*c;
    int i,j,k;
	a=(unsigned char*) malloc(sizeof(unsigned char)*(t*nrc+padding));
	if(a==NULL) {printf("allocu_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(unsigned char**) malloc(sizeof(unsigned char*)*t*nr);
    pp=(unsigned char***) malloc(sizeof(unsigned char**)*t*n);
    ppp=(unsigned char****) malloc(sizeof(unsigned char***)*t);
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            for(j=0;j<r;j++)
                p[k*nr+i*r+j]=&a[k*nrc+i*rc+j*c];
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            pp[k*n+i]=&p[k*nr+i*r];
    for(k=0;k<t;k++)
        ppp[k]=&pp[k*n];
    return(ppp);
}
inline void freeu_4(unsigned char ****p)
{
	if(p!=NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline double**** allocd_4(int t,int n,int r,int c,int padding=DEF_PADDING)
{
	double *a,**p,***pp,****ppp;
    int nrc=n*r*c,nr=n*r,rc=r*c;
    int i,j,k;
	a=(double*) malloc(sizeof(double)*(t*nrc+padding));
	if(a==NULL) {printf("allocd_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(double**) malloc(sizeof(double*)*t*nr);
    pp=(double***) malloc(sizeof(double**)*t*n);
    ppp=(double****) malloc(sizeof(double***)*t);
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            for(j=0;j<r;j++)
                p[k*nr+i*r+j]=&a[k*nrc+i*rc+j*c];
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            pp[k*n+i]=&p[k*nr+i*r];
    for(k=0;k<t;k++)
        ppp[k]=&pp[k*n];
    return(ppp);
}
inline void freed_4(double ****p)
{
	if(p!=NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}

inline float**** allocf_4(int t,int n,int r,int c,int padding=DEF_PADDING)
{
	float *a,**p,***pp,****ppp;
	int nrc=n*r*c,nr=n*r,rc=r*c;
	int i,j,k;
	a=(float*) malloc(sizeof(float)*(t*nrc+padding));
	if(a==NULL) {printf("allocf_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(float**) malloc(sizeof(float*)*t*nr);
	pp=(float***) malloc(sizeof(float**)*t*n);
	ppp=(float****) malloc(sizeof(float***)*t);
	for(k=0;k<t;k++)
		for(i=0;i<n;i++)
			for(j=0;j<r;j++)
				p[k*nr+i*r+j]=&a[k*nrc+i*rc+j*c];
	for(k=0;k<t;k++)
		for(i=0;i<n;i++)
			pp[k*n+i]=&p[k*nr+i*r];
	for(k=0;k<t;k++)
		ppp[k]=&pp[k*n];
	return(ppp);
}
inline void freef_4(float ****p)
{
	if(p!=NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}

void stereo_flip_corr_vol(double***corr_vol_right,double***corr_vol,int h,int w,int nr_plane);
inline void memcpy_u3(unsigned char a[3],unsigned char b[3]){*a++=*b++; *a++=*b++; *a++=*b++;}
inline void image_copy(double***out,double***in,int h,int w,int d){memcpy(out[0][0],in[0][0],sizeof(double)*h*w*d);}
inline void image_copy(unsigned char**out,unsigned char**in,int h,int w){memcpy(out[0],in[0],sizeof(unsigned char)*h*w);}
void depth_best_cost(unsigned char**depth,double***evidence,int h,int w,int nr_planes);
void vec_min_pos(int &min_pos,double *in,int len);
void detect_occlusion_left_right(unsigned char**mask_left,unsigned char**depth_left,unsigned char**depth_right,int h,int w,int nr_plane);
int file_open_ascii(char *file_path,int *out,int len);

//template<class T>
//void SaveZimgJpg(ZImage<T>& zimg, char *filename)
//{
//	CxImage cximg;
//	ZImageToCxImage(zimg,cximg);
//	if(CMDRVideoFrame::s_dScale<1.0)
//		cximg.Resample(CMDRVideoFrame::GetImgWidthSmall(),CMDRVideoFrame::GetImgHeightSmall());
//	char newfilename[30];
//	char* temp=".jpg";
//	sprintf(newfilename,"%s%s",filename,temp);
//	std::string tfilename=CMDRVideoFrame::s_ModelDir+newfilename;
//	cximg.Save(tfilename.c_str(),CXIMAGE_FORMAT_JPG);
//}
#endif