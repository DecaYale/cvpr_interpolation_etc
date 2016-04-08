#include "qx_basic.h"//#include "Nonlocal/Nonlocal_basic.h" //
#include "qx_ppm.h"
#include "qx_hardware_efficient_bilateral_filter.h"
qx_hardware_efficient_bilateral_filter::qx_hardware_efficient_bilateral_filter()
{
	m_sum_value=NULL;
	m_buf_i1=NULL;
	m_textures=NULL;
	m_images=NULL;
	m_images_backup=NULL;
	
	m_image_weight=NULL;
	m_weighted_image=NULL;
}
qx_hardware_efficient_bilateral_filter::~qx_hardware_efficient_bilateral_filter()
{
    clean();
}
void qx_hardware_efficient_bilateral_filter::clean()
{
	if(m_sum_value) delete [] m_sum_value; m_sum_value=NULL;
	qx_freei(m_buf_i1); m_buf_i1=NULL;
	if(m_textures!=NULL)
	{
		for(int s=0;s<m_nr_scale;s++){qx_freeu_3(m_textures[s]); m_textures[s]=NULL;}
		delete [] m_textures; m_textures=NULL;
	}
	if(m_images!=NULL)
	{
		for(int s=0;s<m_nr_scale;s++){qx_freeu_3(m_images[s]); m_images[s]=NULL;}
		delete [] m_images; m_images=NULL;
	}
	if(m_images_backup!=NULL)
	{
		for(int s=0;s<m_nr_scale;s++){qx_freeu_3(m_images_backup[s]); m_images_backup[s]=NULL;}
		delete [] m_images_backup; m_images_backup=NULL;
	}
	if(m_image_weight!=NULL)
	{
		for(int s=0;s<m_nr_scale;s++){qx_freed(m_image_weight[s]); m_image_weight[s]=NULL;}
		delete [] m_image_weight; m_image_weight=NULL;
	}
	if(m_weighted_image!=NULL)
	{
		for(int s=0;s<m_nr_scale;s++){qx_freed_3(m_weighted_image[s]); m_weighted_image[s]=NULL;}
		delete [] m_weighted_image; m_weighted_image=NULL;
	}
}
void qx_hardware_efficient_bilateral_filter::update_weight(int sigma_range)
{
	m_sigma_range=sigma_range; 
	for(int i=0;i<=QX_DEF_CHAR_MAX;i++) m_table[i]=exp(-(double)(i)/(m_sigma_range));
}
int qx_hardware_efficient_bilateral_filter::init(int h,int w,int nr_channel,int nr_scale,int sigma_range,int radius)
{
	clean();
	m_h=h; m_w=w; m_nr_channel=nr_channel; m_nr_scale=nr_scale; m_sigma_range=sigma_range; m_radius=radius;

	m_sum_value=new double [m_nr_channel];
	m_buf_i1=qx_alloci(2,m_nr_scale);

	m_hs=m_buf_i1[0];
	m_ws=m_buf_i1[1];

	m_textures=new unsigned char***[m_nr_scale];
	m_images=new unsigned char***[m_nr_scale];
	m_images_backup=new unsigned char***[m_nr_scale];
	m_image_weight=new double**[m_nr_scale];
	m_weighted_image=new double***[m_nr_scale];
	for(int s=0;s<m_nr_scale;s++)
	{
		if(s==0)
		{
			m_hs[s]=m_h;
			m_ws[s]=m_w;
		}
		else
		{
			m_hs[s]=(m_hs[s-1]>>1);
			m_ws[s]=(m_ws[s-1]>>1);
		}
		m_textures[s]=qx_allocu_3(m_hs[s],m_ws[s],3);
		m_images[s]=qx_allocu_3(m_hs[s],m_ws[s],3);
		m_images_backup[s]=qx_allocu_3(m_hs[s],m_ws[s],3);
		m_image_weight[s]=qx_allocd(m_hs[s],m_ws[s]);
		m_weighted_image[s]=qx_allocd_3(m_hs[s],m_ws[s],3);
		
	}
	m_radius=min(m_radius,int(min(m_hs[m_nr_scale-1],m_ws[m_nr_scale-1])*0.5+0.5));
	update_weight(m_sigma_range);
	return(0);
}

void qx_hardware_efficient_bilateral_filter::down_sample_1(unsigned char ***out,unsigned char ***in,int h,int w,int downsample_rate)
{
	int y,x; int ho,wo; unsigned char *out_y,**in_y,*in_x;
	ho=(h>>downsample_rate); wo=(w>>downsample_rate); 
	out_y=&out[0][0][0];
	for(y=0;y<ho;y++)
	{
		in_y=in[(y<<downsample_rate)];
		for(x=0;x<wo;x++)
		{
			in_x=in_y[(x<<downsample_rate)];
			*out_y++=*in_x++; *out_y++=*in_x++; *out_y++=*in_x++; //R,G,B ¸³Öµ
		}
	}
}
inline unsigned char euro_dist_rgb_max(unsigned char *a,unsigned char *b) {unsigned char x,y,z; x=abs(a[0]-b[0]); y=abs(a[1]-b[1]); z=abs(a[2]-b[2]); return(max(max(x,y),z));}
int qx_hardware_efficient_bilateral_filter::filter(unsigned char***image_filtered,unsigned char***image_in,unsigned char***texture_in)
{
	qx_timer timer;
	timer.start();
	int nr_iter=1;
	
	for(int ii=0;ii<nr_iter;ii++)
	{
		image_copy(m_images[0],image_in,m_h,m_w,m_nr_channel);
		for(int s=0;s<m_nr_scale;s++)
		{
			int h=m_hs[s];
			int w=m_ws[s];
			unsigned char***image_s=m_images[s]; //I
			unsigned char***image_temp_s=m_images_backup[s];
			unsigned char***texture_s=m_textures[s]; // T ?
			if(s==0)
			{
				image_copy(m_textures[s],texture_in,m_hs[s],m_ws[s],3);
			}
			else
			{
				down_sample_1(texture_s,m_textures[s-1],m_hs[s-1],m_ws[s-1],1);
			}
			if(s>0)
			{
				unsigned char***image_larger_resolution=m_images[s-1];
				image_zero(image_s,h,w,m_nr_channel);
				for(int y=0;y<h;y++) for(int x=0;x<w;x++) for(int i=0;i<m_nr_channel;i++) 
				{
					
					int usum=0;
					usum+=image_larger_resolution[(y<<1)][(x<<1)][i];
					usum+=image_larger_resolution[(y<<1)][(x<<1)+1][i];
					usum+=image_larger_resolution[(y<<1)+1][(x<<1)][i];
					usum+=image_larger_resolution[(y<<1)+1][(x<<1)+1][i];
					image_s[y][x][i]=int(usum*0.25+0.5);
				}
			}
		}
		for(int s=m_nr_scale-1;s>=0;s--)
		{
			int h=m_hs[s];
			int w=m_ws[s];
			unsigned char***image_s=m_images[s];
			unsigned char***image_temp_s=m_images_backup[s];
			unsigned char***texture_s=m_textures[s];
			double**image_weight_s=m_image_weight[s];
			double***weighted_image_s=m_weighted_image[s];

			if(s<(m_nr_scale-1))
			{
				int hs=m_hs[s+1];
				int ws=m_ws[s+1];
				int hs1=hs-1;
				int ws1=ws-1;
				unsigned char***image_lower_resolution=m_images[s+1];
				for(int y=0;y<h;y++) for(int x=0;x<w;x++) 
				{
					int ys=(y>>1);
					int xs=(x>>1);
					if(ys<hs&&xs<ws) 
					{
						int dist_rgb=euro_dist_rgb_max(m_textures[s][y][x],m_textures[s+1][ys][xs]); 
						double weight_for_lower_resolution=m_table[dist_rgb];   //alpha ?
						double weight_for_larger_resolution=1-weight_for_lower_resolution;//1-alpha
						for(int i=0;i<m_nr_channel;i++) 
						{
							double sum_weight=weight_for_larger_resolution+weight_for_lower_resolution*m_image_weight[s+1][ys][xs];
							double sum_value=image_s[y][x][i]*weight_for_larger_resolution+weight_for_lower_resolution*m_weighted_image[s+1][ys][xs][i];
							image_s[y][x][i]=int(sum_value/sum_weight+0.5);
						}
					}
				}
			}
			filter(weighted_image_s,image_weight_s,image_s,texture_s,h,w,m_nr_channel,m_radius);
		}
	}
	for(int y=0;y<m_h;y++) for(int x=0;x<m_w;x++) for(int i=0;i<m_nr_channel;i++) image_filtered[y][x][i]=int(m_weighted_image[0][y][x][i]/m_image_weight[0][y][x]+0.5);
    return(0);
}
void qx_hardware_efficient_bilateral_filter::filter(double***weighted_image,double**image_weight,unsigned char***image,unsigned char***texture,int h,int w,int nr_channel,int radius)
{
	for(int y=0;y<h;y++)
	{
		int ymin=max(y-radius,0); 
		int ymax=min(y+radius,h-1); 
		for(int x=0;x<w;x++) 
		{
			int xmin=max(x-radius,0); 
			int xmax=min(x+radius,w-1);
			double*weighted_image_=weighted_image[y][x];
			unsigned char*texture_yx=texture[y][x];

			memset(m_sum_value,0,sizeof(double)*nr_channel);
			double sum_weight=0;

			for(int yi=ymin;yi<=ymax;yi++) //´°¿Ú
			{
				unsigned char*texture_yxi=texture[yi][xmin];
				unsigned char*image_yxi=image[yi][xmin];
				for(int xi=xmin;xi<=xmax;xi++)
				{
					double weight=m_table[euro_dist_rgb_max(texture_yxi,texture_yx)];
					*texture_yxi++; *texture_yxi++; *texture_yxi++; //¿ç¹ýRGB
					double*sum_value=m_sum_value;
					for(int d=0;d<nr_channel;d++) (*sum_value++)+=(*image_yxi++)*weight;
					sum_weight+=weight;
				}
			}			
			image_weight[y][x]=sum_weight;
			for(int d=0;d<nr_channel;d++) weighted_image_[d]=m_sum_value[d];
		}
	}
}
