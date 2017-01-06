#ifndef CU_DEM_DEC_CUH
#define CU_DEM_DEC_CUH

#include <vector_types.h>
#include <helper_math.h>
//#include <thrust/transform.h>
// #include <thrust/device_vector.h>
// #include <thrust/generate.h>
//#include <thrust/functional.h>
// #include <thrust/sort.h>
// #include <thrust/binary_search.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/random.h>
//#include <iterator>

//typedef thrust::tuple<double, double, double> vec3;

// struct summation_vec3 : public thrust::unary_function<vec3, vec3>
// {
// 	__host__ __device__
// 		vec3 operator()(const vec3& a) const {
// 			return 
// 		}
// };
// struct summation_double3
// {
// 	int a;
// 	summation_double3(const int _a) : a(_a) {}
// 	__host__ __device__
// 		double operator()(const double v1) const {
// 			double v = 0;
// 			switch(a){
// 				case 0: v = v1.x; break;
// 				case 1: v = v1.y; break;
// 				case 2: v = v1.z; break;
// 			}
// 			return v;
// 		}
// };

struct device_parameters
{
	unsigned int np;
	unsigned int nsp;
	unsigned int ncell;
	uint3 grid_size;
	double dt;
	double half2dt;
	double cell_size;
	double3 gravity;
	double3 world_origin;
	
	double kn;
	double vn;
	double ks;
	double vs;
	double mu;

	double cohesive;
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
};

struct cu_polygon
{
	double3 P;
	double3 Q;
	double3 R;
	double3 V;
	double3 W;
	double3 N;
};

struct cu_mass_info
{
	double3 pos;
	double3 vel;
};

struct cu_contact_info
{
	double penetration;
	double3 unit;
};

void setSymbolicParameter(device_parameters *h_paras);

// velocity-verlet integration functions
void vv_update_position(double *pos, double *vel, double *acc, unsigned int np);
void vv_update_velocity(double *vel, double *acc, double *omega, double *alpha, double *force, double *moment, unsigned int np);

// cell grid detection functions
void cu_calculateHashAndIndex(unsigned int* hash, unsigned int* index, double *pos, unsigned int np);
void cu_reorderDataAndFindCellStart(unsigned int* hash, unsigned int* index, unsigned int* cstart, unsigned int* cend, unsigned int* sorted_index, unsigned int np, unsigned int nc);

// dem force function
void cu_calculate_p2p(double* pos, double* vel, double* acc, double* omega, double* alpha, double* force, double* moment, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int np, unsigned int cRun = 0);

// geometry contact force functions
void cu_cube_hertzian_contact_force(device_plane_info* plans, double kn, double vn, double ks, double vs, double mu, bool* isLineContact, double* pos, double* vel, double* omega, double* force, double* moment, unsigned int np);
double3 cu_shape_hertzian_contact_force(cu_mass_info *minfo, cu_polygon* polygons, unsigned int *id_set, unsigned int *poly_start, unsigned int *poly_end, double kn, double vn, double ks, double vs, double mu, bool* isLineContact, double* pos, double* vel, double* omega, double* force, double* moment, double3* bforce, unsigned int np, unsigned int* sorted_id, unsigned int* cell_start, unsigned int* cell_end);
void cu_update_polygons_position(double3* mpos, double* A, double3* local_vertice, unsigned int nvertex, double3* pos);
void cu_update_polygons_information(cu_polygon* polygons, uint3 *indice, double3* pos, unsigned int ni);
void cu_mergedata(double* data, double* pos, uint np, double3* vertice, uint snp);



#endif