/***************************************************************************************
\Author:	Qingxiong Yang
\Function:	(Joint/Cross) Bilateral Filtering.
\Reference:	Qingxiong Yang, Hardware-Efficient Bilateral Filtering for Stereo Matching, 
			IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2014.
****************************************************************************************/
#ifndef QX_HARDWARE_EFFICIENT_BILATERAL_FILTER_H
#define QX_HARDWARE_EFFICIENT_BILATERAL_FILTER_H

class qx_hardware_efficient_bilateral_filter
{
public:
    qx_hardware_efficient_bilateral_filter();
    ~qx_hardware_efficient_bilateral_filter();
    void clean();
	int init(int h,int w,int nr_channel,
		int nr_scale=6,//control spatial filter kerner, similar to sigma_spatial
		int sigma_range=10,//control range filter kernel
		int radius=2//can be as small as 1
		);
    int filter(unsigned char***image_filtered,unsigned char***image_in,unsigned char***texture_in);
	int filter_approximate(unsigned char***image_filtered,unsigned char***image_in,unsigned char***texture_in);
private:
	void update_weight(int sigma_range);
	void filter(double***weighted_image,double**image_weight,unsigned char***image,unsigned char***texture,int h,int w,int nr_channel,int radius);
	void down_sample_1(unsigned char ***out,unsigned char ***in,int h,int w,int downsample_rate);
private:
	int	m_h,m_w,m_nr_channel,m_nr_scale,m_sigma_range,m_radius; 
	int*m_hs,*m_ws;
	int**m_buf_i1;
	unsigned char****m_textures,****m_images,****m_images_backup;
	double***m_image_weight;
	double****m_weighted_image;
	double m_table[QX_DEF_CHAR_MAX+1]; 
	double*m_sum_value;
};
#endif