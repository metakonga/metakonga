#include "mphysics_cuda_dec.cuh"
#include "mphysics_cuda_impl.cuh"
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
//#include <helper_cuda.h>

void setSymbolicParameter(device_parameters *h_paras)
{
	checkCudaErrors(cudaMemcpyToSymbol(cte, h_paras, sizeof(device_parameters)));
}

unsigned int numThreads, numBlocks;

//Round a / b to nearest higher integer value
unsigned iDivUp(unsigned a, unsigned b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(unsigned n, unsigned blockSize, unsigned &numBlocks, unsigned &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

void vv_update_position(double *pos, double *vel, double *acc, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	vv_update_position_kernel <<< numBlocks, numThreads >>>(
		(double4 *)pos,
		(double3 *)vel,
		(double3 *)acc);
}

void vv_update_velocity(double *vel, double *acc, double *omega, double *alpha, double *force, double *moment, double* mass, double* iner, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	vv_update_velocity_kernel <<< numBlocks, numThreads >>>(
		(double3 *)vel,
		(double3 *)acc,
		(double3 *)omega,
		(double3 *)alpha,
		(double3 *)force,
		(double3 *)moment,
		mass,
		iner);
}

void cu_calculateHashAndIndex(
	unsigned int* hash,
	unsigned int* index,
	double *pos,
	unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	calculateHashAndIndex_kernel <<< numBlocks, numThreads >>>(hash, index, (double4 *)pos);


}

void cu_calculateHashAndIndexForPolygonSphere(
	unsigned int* hash,
	unsigned int* index,
	unsigned int sid,
	unsigned int nsphere,
	double *sphere)
{
	computeGridSize(nsphere, 512, numBlocks, numThreads);
	calculateHashAndIndexForPolygonSphere_kernel << <numBlocks, numThreads >> >(hash, index, sid, nsphere, (double4 *)sphere);
}


void cu_reorderDataAndFindCellStart(
	unsigned int* hash,
	unsigned int* index,
	unsigned int* cstart,
	unsigned int* cend,
	unsigned int* sorted_index,
	unsigned int np,
	unsigned int nsphere,
	unsigned int ncell)
{
	//std::cout << "step 1" << std::endl;
	unsigned int tnp = np + nsphere;
	thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),
		thrust::device_ptr<unsigned>(hash + tnp),
		thrust::device_ptr<unsigned>(index));
	//std::cout << "step 2" << std::endl;
	computeGridSize(tnp, 256, numBlocks, numThreads);
	checkCudaErrors(cudaMemset(cstart, 0xffffffff, ncell*sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(cend, 0, ncell*sizeof(unsigned int)));
	unsigned smemSize = sizeof(unsigned int)*(numThreads + 1);
	//std::cout << "step 3" << std::endl;
	reorderDataAndFindCellStart_kernel <<< numBlocks, numThreads, smemSize >>>(
		hash,
		index,
		cstart,
		cend,
		sorted_index);
//  	std::fstream fs;
//  	fs.open("C:/C++/gpu_cstart_cend.txt", std::ios::out);
//  	unsigned int *h_start = new unsigned int[nc];
//  	unsigned int *h_end = new unsigned int[nc];
//  	checkCudaErrors(cudaMemcpy(h_start, cstart, sizeof(unsigned int)*nc, cudaMemcpyDeviceToHost));
//  	checkCudaErrors(cudaMemcpy(h_end, cend, sizeof(unsigned int)*nc, cudaMemcpyDeviceToHost));
// 
// 	for (unsigned int i = 0; i < nc; i++){
// 		fs << h_start[i] << " " << h_end[i] << std::endl;
// 	}
// 	delete[] h_start;
// 	delete[] h_end;
//  	fs.close();
	//std::cout << "step 4" << std::endl;
}

void cu_calculate_p2p(
	const int tcm, double* pos, double* vel, double* acc, 
	double* omega, double* alpha, 
	double* force, double* moment, 
	double* mass, double* iner,
	unsigned int* sorted_index, unsigned int* cstart, 
	unsigned int* cend, contact_parameter* cp, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	switch (tcm)
	{
	case 0:
		calculate_p2p_kernel<0> << < numBlocks, numThreads >> >(
			(double4 *)pos, (double3 *)vel, (double3 *)acc,
			(double3 *)omega, (double3 *)alpha,
			(double3 *)force, (double3 *)moment,
			mass, iner,
			sorted_index, cstart,
			cend, cp);
		break;
	case 1:
		calculate_p2p_kernel<1> << < numBlocks, numThreads >> >(
			(double4 *)pos, (double3 *)vel, (double3 *)acc,
			(double3 *)omega, (double3 *)alpha,
			(double3 *)force, (double3 *)moment,
			mass, iner,
			sorted_index, cstart,
			cend, cp);
		break;
	}
}

void cu_plane_hertzian_contact_force(
	const int tcm, device_plane_info* plan, 
	double* pos, double* vel, double* omega, 
	double* force, double* moment, double* mass, 
	unsigned int np, contact_parameter *cp)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	switch (tcm)
	{
	case 0: plane_hertzian_contact_force_kernel<0> << < numBlocks, numThreads >> >(
		plan, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass);
		break;
	case 1: plane_hertzian_contact_force_kernel<1> << < numBlocks, numThreads >> >(
		plan, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass);
		break;
	}
	
}

void cu_cylinder_hertzian_contact_force(
	const int tcm, device_cylinder_info* cyl, 
	double* pos, double* vel, double* omega, 
	double* force, double* moment, 
	double* mass, unsigned int np, contact_parameter *cp,
	double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	switch (tcm)
	{
	case 0: cylinder_hertzian_contact_force_kernel<0> << < numBlocks, numThreads >> >(
		cyl, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass, mpos, mf, mm);
		break;
	case 1: cylinder_hertzian_contact_force_kernel<1> << < numBlocks, numThreads >> >(
		cyl, (double4 *)pos, (double3 *)vel, (double3 *)omega,
		(double3 *)force, (double3 *)moment, cp, mass, mpos, mf, mm);
		break;
	}
	
}

void cu_particle_polygonObject_collision(
	const int tcm, device_polygon_info* dpi, double* dsph, device_polygon_mass_info* dpmi, 
	double* pos, double* vel, double* omega, 
	double* force, double* moment, double* mass, 
	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, contact_parameter *cp,
	unsigned int np, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	switch (tcm)
	{
	case 0: 
		particle_polygonObject_collision_kernel<0> << < numBlocks, numThreads >> >(
		dpi, (double4 *)dsph, dpmi, 
		(double4 *)pos, (double3 *)vel, (double3 *)omega, 
		(double3 *)force, (double3 *)moment, mass, 
		sorted_index, cstart, cend, cp,
		mpos, mf, mm);
		break;

	}
	
}

double3 reductionD3(double3* in, unsigned int np)
{
	double3 rt = make_double3(0.0, 0.0, 0.0);
	computeGridSize(np, 512, numBlocks, numThreads);
	double3* d_out;
	double3* h_out = new double3[numBlocks];
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(double3) * numBlocks));
	checkCudaErrors(cudaMemset(d_out, 0, sizeof(double3) * numBlocks));
	//unsigned smemSize = sizeof(double3)*(512);
	reduce6<double3, 512> << < numBlocks, numThreads/*, smemSize*/ >> >(in, d_out, np);
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
	for (unsigned int i = 0; i < numBlocks; i++){
		rt.x += h_out[i].x;
		rt.y += h_out[i].y;
		rt.z += h_out[i].z;
	}
	delete[] h_out;
	checkCudaErrors(cudaFree(d_out));
	return rt;
}