#ifndef TREE_FILTER_H
#define TREE_FILTER_H
#include "mst_kruskals_image.h"
#include "Nonlocal/Nonlocal_basic.h"
class tree_filter
{
public:
	tree_filter();
	~tree_filter();
	void clean();
	int init(int h,int w,int nr_channel,double sigma_range=DEF_MST_KI_SIGMA_RANGE,int nr_neighbor=DEF_MST_KI_4NR_NEIGHBOR);
	int build_tree(unsigned char*texture);
	//void init_tree_value(unsigned char*image);
	template<typename T>void init_tree_value(T*image,bool compute_weight);
	template<typename T>void combine_tree(T*image_filtered);
	int filter(float*cost,float*cost_backup,int nr_plane);
	int*get_rank(){return(m_mst_rank);};
	void update_table(double sigma_range);
private:
	mst_kruskals_image m_mst;
	int m_h,m_w,m_nr_channel; int m_nr_pixel;
	int*m_mst_parent;
	int*m_mst_nr_child;
	int**m_mst_children;//[DEF_MST_NODE_MAX_NR_CHILDREN];
	int*m_mst_rank;
	unsigned char*m_mst_weight;//cost between this node and its parent

	double m_table[DEF_CHAR_MAX+1];
	int*m_node_id;
private:
	void filter_main(bool compute_weight);
};
void test_tree_filter();

#endif