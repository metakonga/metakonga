#ifndef MPHYSICS_CUDA_DEC_CUH
#define MPHYSICS_CUDA_DEC_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>
//#include <helper_math.h>

//#include <helper_functions.h>
//#include <helper_cuda.h>

//__constant__ device_parameters cte;
//double3 toDouble3(VEC3D& v3) { return double3(v3.x, v3.y, v3.z); }

struct device_parameters
{
	unsigned int np;
	unsigned int nsphere;
	unsigned int ncell;
	uint3 grid_size;
	double dt;
	double half2dt;
	double cell_size;
	double cohesion;
	double3 gravity;
	double3 world_origin;

	//float cohesive;
};

struct device_polygon_info
{
	int id;
	double3 P;
	double3 Q;
	double3 R;
	double3 V;
	double3 W;
	double3 N;
};

struct device_plane_info
{
	double l1, l2;
	double3 u1;
	double3 u2;
	double3 uw;
	double3 xw;
	double3 pa;
	double3 pb;
	double3 w2;
	double3 w3;
	double3 w4;
};

struct device_polygon_mass_info
{
	double3 origin;
	double4 ep;
	double3 vel;
	double3 omega;
	double3 force;
	double3 moment;
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

struct device_contact_property
{
	double Ei, Ej;
	double pri, prj;
	double Gi, Gj;
	double rest;
	double fric;
	double rfric;
	double coh;
	double sratio;
};


struct device_force_constant
{
	double kn;
	double vn;
	double ks;
	double vs;
	double mu;
	double ms;
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

void vv_update_position(double *pos, double *vel, double *acc, unsigned int np);
void vv_update_velocity(double *vel, double *acc, double *omega, double *alpha, double *force, double *moment, double* mass, double* iner, unsigned int np);

void cu_calculateHashAndIndex(unsigned int* hash, unsigned int* index, double *pos, unsigned int np);
void cu_calculateHashAndIndexForPolygonSphere(unsigned int* hash, unsigned int* index, unsigned int sid, unsigned int nsphere, double *sphere);
void cu_reorderDataAndFindCellStart(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index, unsigned int np, /*unsigned int nsphere,*/ unsigned int ncell);

void cu_calculate_p2p(
	const int tcm, double* pos, double* vel, 
	double* omega, double* force,
	double* moment, double* mass, 
	unsigned int* sorted_index, unsigned int* cstart, 
	unsigned int* cend, device_contact_property* cp, unsigned int np);

// Function for contact between particle and plane
void cu_plane_contact_force(
	const int tcm, device_plane_info* plan, 
	double* pos, double* vel, double* omega, 
	double* force, double* moment, double* mass, 
	unsigned int np, device_contact_property *cp);

void cu_cube_contact_force(
	const int tcm, device_plane_info* plan,
	double* pos, double* vel, double* omega,
	double* force, double* moment, double* mass,
	unsigned int np, device_contact_property *cp);

// Function for contact between particle and polygonObject
void cu_particle_polygonObject_collision(
	const int tcm, device_polygon_info* dpi, double* dsph, device_polygon_mass_info* dpmi,
	double* pos, double* vel, double* omega, 
	double* force, double* moment, double* mass, 
	unsigned int* sidx, unsigned int* cstart, unsigned int* cend, device_contact_property *cp,
	unsigned int np/*, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm*/);

// Function for contact between particle and cylinder
void cu_cylinder_hertzian_contact_force(
	const int tcm, device_cylinder_info* cyl, 
	double* pos, double* vel, double* omega, 
	double* force, double* moment, double* mass, 
	unsigned int np, device_contact_property* cp,
	double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm);

double3 reductionD3(double3* in, unsigned int np);

#endif



