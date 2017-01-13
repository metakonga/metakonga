#ifndef MPHYSICS_CUDA_DEC_CUH
#define MPHYSICS_CUDA_DEC_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>
//#include <helper_math.h>

//#include <helper_functions.h>
//#include <helper_cuda.h>

//__constant__ device_parameters cte;

struct device_parameters
{
	unsigned int np;
	unsigned int nsphere;
	unsigned int ncell;
	uint3 grid_size;
	float dt;
	float half2dt;
	float cell_size;
	float cohesion;
	float3 gravity;
	float3 world_origin;

	//float cohesive;
};

struct device_polygon_info
{
	double3 P;
	double3 Q;
	double3 R;
	double3 V;
	double3 W;
	double3 N;
};

struct device_plane_info
{
	float l1, l2;
	float3 u1;
	float3 u2;
	float3 uw;
	float3 xw;
	float3 pa;
	float3 pb;
	float3 w2;
	float3 w3;
	float3 w4;
};

struct device_polygon_mass_info
{
	double3 origin;
	double3 vel;
	double3 omega;
	double4 ep;
};

struct device_cylinder_info
{
	double len, rbase, rtop;
	double3 pbase;
	double3 ptop;
	double3 origin;
	double3 vel;
	double3 omega;
	double4 ep;
};

template<typename T>
struct device_force_constant
{
	T kn;
	T vn;
	T ks;
	T vs;
	T mu;
};

struct device_force_constant_d
{
	double kn;
	double vn;
	double ks;
	double vs;
	double mu;
};

void setSymbolicParameter(device_parameters *h_paras);

void vv_update_position(float *pos, float *vel, float *acc, unsigned int np);
void vv_update_velocity(float *vel, float *acc, float *omega, float *alpha, float *force, float *moment, float* mass, float* iner, unsigned int np);

void cu_calculateHashAndIndex(unsigned int* hash, unsigned int* index, float *pos, unsigned int np);
void cu_calculateHashAndIndexForPolygonSphere(unsigned int* hash, unsigned int* index, unsigned int sid, unsigned int nsphere, double *sphere);
void cu_reorderDataAndFindCellStart(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index, unsigned int np, unsigned int nsphere, unsigned int ncell);

void cu_calculate_p2p(float* pos, float* vel, float* acc, float* omega, float* alpha, float* force, float* moment, float* mass, float* iner, float* riv, float E, float pr, float rest, float sh, float fric, float rfric, float coh, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int np, unsigned int cRun = 0);
void cu_plane_hertzian_contact_force(const int tcm, device_plane_info* plan, float E, float pr, float G, float rest, float fric, float rfric, float* pos, float* vel, float* omega, float* force, float* moment, float* mass, float pE, float pPr, float pG, unsigned int np);
void cu_particle_polygonObject_collision(const int tcm, device_polygon_info* dpi, double* dsph, device_polygon_mass_info* dpmi, float E, float pr, float G, float rest, float fric, float rfric, float* pos, float* vel, float* omega, float* force, float* moment, float* mass, float pE, float pPr, float pG, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int np, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm);
void cu_cylinder_hertzian_contact_force(const int tcm, device_cylinder_info* cyl, float E, float pr, float G, float rest, float fric, float rfric, float* pos, float* vel, float* omega, float* force, float* moment, float* mass, float pE, float pPr, float pG, unsigned int np, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm);
double3 reductionD3(double3* in, unsigned int np);

#endif



