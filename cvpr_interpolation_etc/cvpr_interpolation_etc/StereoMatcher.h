#pragma once

#include "cxcore.h"

extern void RawCost(const cv::Mat & img_L,const cv::Mat & img_R, cv::Mat & dataCostCube,int dLevels);
extern void costAggregate(cv::Mat & winCostCube ,cv::Mat & dataCostCube,int height,int width, int dLevels,int winWidth);

extern void WTA(const cv::Mat & img_L,const cv::Mat & img_R,cv::Mat & depthMap,int dLevels ,int winWidth);


//class CStereoMatcher
//{
//public:
//	void ComputeCorrespondence();
//	// Invoke all of the stereo matching processing steps
//	void Evaluate();
//	// Evaluate the quality of the matching score
//
//protected:
//	void PreProcess();
//	// Pre-process the images to clean them up or normalize them
//	void RawCosts();
//	// Compute raw per-pixel matching score between a pair of frames
//	void Aggregate();
//	// Use spatial aggregation to improve the matching scores
//	void Optimize();
//	// Select the best matches using local or global optimization
//	void Refine();
//	// Refine the matching disparity to get a sub-pixel match
//
//	// Aggregation functions:
//	void AggrBox();                 // box filter
//	void AggrDiffusion(int iter);   // regular and membrane diffusion
//	void AggrBayesian(int iter);    // Bayesian diffusion
//	void AggrMin();                 // min filter (shiftable windows)
//
//	void PadCosts();                // fill the outside parts of the DSI
//	void AggrSubPixelFit();         // fit quadratics to local minima
//	void AggrCollapse();            // fit quadratics to local minima
//
//	// Energy computation functions:
//	void ComputeSmoothnessCosts(); // set up smoothness costs for DP and graph cut
//	void ComputeEnergy(float& denergy, float& nenergy);
//
//	// Optimization functions:
//	void OptWTA();      // simple minimization (winner takes all)
//	void OptDP();       // dynamic programming stereo (Intille and Bobick, no GCPs)
//	void OptSO();       // scanline optimization
//	void OptGraphCut(); // graph cuts
//	void OptSimulAnnl();// simulated annealing
//	void OptSymmetric();// symmetric winner-take-all *NEW*
//
//	// Evaluation functions
//	//CByteImage ComputeOcclusion(int frame);  // compute an occlusion map
//	void ComputeOcclusions();           // compute occlusion map w.r.t. match
//	void ComputeTextureless();          // compute textureless regions
//	void ComputeDisparityDiscont();     // compute disparity discontinuities
//	void ComputeDisparityErrors();      // compute disparity errors
//	void ComputePredictionErrors();     // compute frame prediction errors
//	///void ComputePredictionError(CByteImage& predicted, CByteImage& original,
//	///	float& rms_error, float& fraction_visible);
//	void ComputeMatchQuality();         // compute final matching cost and certainty
//	void ComputeStatusErrors();         // compute disparity errors grouped by status
//
//
//	// Utility functions, shared by several compilation units
//	void StartTiming();                 // start timing evaluation
//	void PrintTiming();                 // print the elapsed time
//
//	//static void WriteCosts(CFloatImage& cost, char *filename);
//public:
//	///static void ComputeEnergy(CFloatImage& dcost, CFloatImage& ncost, 
//		///CIntImage& label, float& denergy, float& nenergy);
//	///static void DumpDisparity(CIntImage& disp, const char* filename, float scale);
//
//protected:
//	float m_disp_step;                  // disparity step size (after possible collapse)
//	int m_disp_n;                       // number of disparity levels (after possible collapse)
//	int m_disp_num;                     // numerator of disparity step size
//	int m_disp_den;                     // denominator of disparity step size
//	float m_disp_step_inv;              // inverse of disp_step (accurate)
//	CByteImage m_reference;             // reference image
//	CByteImage m_matching;              // matching (target) image
//	int m_frame_diff;                   // frame difference between ref and match
//	int m_frame_diff_sign;              // sign of frame difference (for positive disparities)
//	float m_match_outside;              // artificial cost for a match outside boundary
//	CFloatImage m_cost;                 // raw matching costs (# bands = # disparities)
//	CFloatImage m_cost2;                // second cost array - storage used by some algs
//	CFloatImage m_cost0;                // initial cost array - storage used by some algs
//	CFloatImage m_sub_pixel_min;        // location of lowest interpolated value
//	CFloatImage m_sub_pixel_cert;       // certainty in lowest interpolated value
//	CFloatImage m_final_cost;           // best (lowest) cost at winning disparity
//	CFloatImage m_certainty;            // certainty (inv. variance) at winning disparity
//	CByteImage m_status;                // status of matched pixels (EStereoMatchStatus)
//	CByteImage m_status_disp;           // combined status a disparity color image
//	CFloatImage m_smooth;               // spatially varying smoothness weights (2 band)
//	std::vector<float> m_rho_s;         // table of smoothness penalties vs. disp difference
//	CFloatImage m_prob;                 // probability estimates (mean-field)
//	CIntImage m_disparity;              // winning disparities [0..n_disp-1]
//	CFloatImage m_float_disparity;      // sub-pixel disparities [disp_min..disp_max]
//	CFloatImage m_true_disparity;       // ground truth
//	CByteImage m_disparity_error;       // scaled error in disparities
//	CByteImage m_bad_pixels;            // pixels flagged with bad disparities
//	CByteImage m_disparity_histogram;   // histogram of computed disparities
//	CByteImage m_occlusion;             // occlusion map (computed internally)
//	CByteImage m_textureless;           // textureless regions (computed internally)
//	CByteImage m_depth_discont;         // depth discontinuities (computed internally)
//	clock_t m_start_time;               // start of elapsed time interval
//	float m_elapsed_time;               // elapsed time in seconds
//
//	std::vector<CStereoFrame> m_frame;  // input/output images
//};