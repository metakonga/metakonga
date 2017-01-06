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

void vv_update_position(float *pos, float *vel, float *acc, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	vv_update_position_kernel <<< numBlocks, numThreads >>>(
		(float4 *)pos,
		(float3 *)vel,
		(float3 *)acc);
}

void vv_update_velocity(float *vel, float *acc, float *omega, float *alpha, float *force, float *moment, float* mass, float* iner, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	vv_update_velocity_kernel <<< numBlocks, numThreads >>>(
		(float3 *)vel,
		(float3 *)acc,
		(float3 *)omega,
		(float3 *)alpha,
		(float3 *)force,
		(float3 *)moment,
		mass,
		iner);
}

void cu_calculateHashAndIndex(
	unsigned int* hash,
	unsigned int* index,
	float *pos,
	unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	calculateHashAndIndex_kernel <<< numBlocks, numThreads >>>(hash, index, (float4 *)pos);


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

void cu_calculate_p2p(float* pos, float* vel, float* acc, float* omega, float* alpha, float* force, float* moment, float* mass, float* iner, float* riv, float E, float pr, float rest, float ratio, float fric, float coh, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int np, unsigned int cRun)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	calculate_p2p_kernel <<< numBlocks, numThreads >>>(
		(float4 *)pos,
		(float3 *)vel,
		(float3 *)acc,
		(float3 *)omega,
		(float3 *)alpha,
		(float3 *)force,
		(float3 *)moment,
		mass,
		iner,
		riv,
		E,
		pr,
		rest,
		ratio,
		fric,
		coh,
		sorted_index,
		cstart,
		cend,
		cRun);
}

void cu_plane_hertzian_contact_force(float* riv, device_plane_info* plan, float E, float pr, float rest, float ratio, float fric, float* pos, float* vel, float* omega, float* force, float* moment, float* mass, float pE, float pPr, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	plane_hertzian_contact_force_kernel << < numBlocks, numThreads >> >(
		plan,
		E,
		pr,
		rest,
		ratio,
		fric,
		(float4 *)pos,
		(float3 *)vel,
		(float3 *)omega,
		(float3 *)force,
		(float3 *)moment,
		mass,
		riv,
		pE,
		pPr);
}

void cu_cylinder_hertzian_contact_force(device_cylinder_info* cyl, float E, float pr, float rest, float ratio, float fric, float* pos, float* vel, float* omega, float* force, float* moment, float* mass, float pE, float pPr, unsigned int np, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm)
{
	computeGridSize(np, 512, numBlocks, numThreads);
	cylinder_hertzian_contact_force_kernel << < numBlocks, numThreads >> >(
		cyl,
		E,
		pr,
		rest,
		ratio,
		fric,
		(float4 *)pos,
		(float3 *)vel,
		(float3 *)omega,
		(float3 *)force,
		(float3 *)moment,
		mass,
		pE,
		pPr,
		mpos,
		mf,
		mm);
// 	double3 *d_result_bf;
// 	double3 *d_result_bm;
// 
// 	//double3 *result_bm;
// 	//double3* h_data = new double3[10];
// 	double3* h_result_bf = new double3[numBlocks];
// 	double3* h_result_bm = new double3[numBlocks];
// // 	for (unsigned int i = 0; i < 10; i++){
// // 		h_data[i] = make_double3(i * 1, i * 2, i * 3);
// // 	}
// 	//double3* h_result_bm = new double3[numBlocks];
// 	double3* h_mf = new double3[np];
// 	//double3* d_data;
// 	//checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(double3) * 10));
// 	//checkCudaErrors(cudaMemcpy(d_data, h_data, sizeof(double3) * 10, cudaMemcpyHostToDevice));
// 	memset(h_result_bf, 0, sizeof(double3) * numBlocks); 
// 	memset(h_result_bm, 0, sizeof(double3) * numBlocks);
// 	//memset(h_result_bm, 0, sizeof(double3) * numBlocks);
// 	checkCudaErrors(cudaMalloc((void**)&d_result_bf, sizeof(double3) * numBlocks));
// 	checkCudaErrors(cudaMalloc((void**)&d_result_bm, sizeof(double3) * numBlocks));
// 	//checkCudaErrors(cudaMalloc((void**)&result_bm, sizeof(double3) * numBlocks));
// 	checkCudaErrors(cudaMemset(d_result_bf, 0, sizeof(double3) * numBlocks));
// 	checkCudaErrors(cudaMemset(d_result_bm, 0, sizeof(double3) * numBlocks));
// 	//checkCudaErrors(cudaMemset(result_bm, 0, sizeof(double3)*numBlocks));
// 	//double3 tmf = thrust::reduce(thrust::device_ptr<double3>(mf), thrust::device_ptr<double3>(mf) +np);
// 	//unsigned smemSize = sizeof(double3)*(512);
// 	//int data[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
// 	//int *d_data;
// 	////checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(int) * 10));
// 	//checkCudaErrors(cudaMemcpy(d_data, data, sizeof(int) * 10, cudaMemcpyHostToDevice));
// 	//int* d_rst;
// 	//checkCudaErrors(cudaMalloc((void**)&d_rst, sizeof(int) * numBlocks));
// 	//smemSize = sizeof(int) * 256;
// 	//reduce6<int, 256> << < numBlocks, 10, smemSize >> >(d_data, d_rst, 10);
// 	//int* h_rst = new int[numBlocks];
// 	//checkCudaErrors(cudaMemcpy(h_rst, d_rst, sizeof(int)*numBlocks, cudaMemcpyDeviceToHost));
// // 	reduce6<double3, 512> << < numBlocks, numThreads, smemSize >> >(mf, d_result_bf, np);
// // 	//reduce6<double3, 256> <<< numBlocks, numThreads, smemSize >>>(mm, result_bm, np);
// // 	
// // 	
// //  	checkCudaErrors(cudaMemcpy(h_result_bf, d_result_bf, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
// // 	for (unsigned int i = 0; i < numBlocks; i++){
// // 		_mf.x += h_result_bf[i].x;
// // 		_mf.y += h_result_bf[i].y;
// // 		_mf.z += h_result_bf[i].z;
// // 	}
//  //	checkCudaErrors(cudaMemset(d_result_bf, 0, sizeof(double3) * numBlocks));
// //  	reduce6<double3, 512> << < numBlocks, numThreads, smemSize >> >(mm, d_result_bm, np);
// // 	checkCudaErrors(cudaMemcpy(h_result_bm, d_result_bm, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
// // 	for (unsigned int i = 0; i < numBlocks; i++){
// // 		_mm.x += h_result_bm[i].x;
// // 		_mm.y += h_result_bm[i].y;
// // 		_mm.z += h_result_bm[i].z;
// // 	}
// 	//checkCudaErrors(cudaMemcpy(h_result_bm, result_bm, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost));
// 	checkCudaErrors(cudaMemcpy(h_mf, mf, sizeof(double3)*np, cudaMemcpyDeviceToHost));
// 	for (unsigned int i = 0; i < np; i++){
// 		_mf += h_mf[i];
// 	}
// // 	double3 sum_d = make_double3(0.0, 0.0, 0.0);
// // 	for (unsigned int i = 0; i < numBlocks; i++){
// // 		sum_d.x += h_result_bf[i].x;
// // 		sum_d.y += h_result_bf[i].y;
// // 		sum_d.z += h_result_bf[i].z;
// // 	}
// // 	
//  	checkCudaErrors(cudaMemcpy(h_mf, mm, sizeof(double3)*np, cudaMemcpyDeviceToHost));
// 	for (unsigned int i = 0; i < np; i++){
// 		_mm += h_mf[i];
// 	}
// 	
// // 	for (unsigned int i = 0; i < numBlocks; i++){
// // 		_mf.x += h_result_bf[i].x;
// // 		_mf.y += h_result_bf[i].y;
// // 		//std::cout << _mf.x << std::endl;
// // 		_mf.z += h_result_bf[i].z;
// // // 		_mm.x += h_result_bm[i].x;
// // // 		_mm.y += h_result_bm[i].y;
// // // 		_mm.z += h_result_bm[i].z;
// // 	}
// // 	if (sum_d.y != 0){
// // 		std::cout << "sum_d = " << sum_d.x << " " << sum_d.y << " " << sum_d.z << std::endl;
// // 		std::cout << "_mf = " << _mf.x << " " << _mf.y << " " << _mf.z << std::endl;
// // 	}
// 	//delete[] h_data;
// 	delete[] h_mf;
//  	delete[] h_result_bf;
//  	delete[] h_result_bm;
// // 	// 	//delete [] h_force;¤±`1																																			¤Ä{¤²1
//  	checkCudaErrors(cudaFree(d_result_bf));
//  	checkCudaErrors(cudaFree(d_result_bm));
}

void cu_particle_polygonObject_collision(device_polygon_info* dpi, double* dsph, device_polygon_mass_info* dpmi, float E, float pr, float rest, float ratio, float fric, float* pos, float* vel, float* omega, float* force, float* moment, float* mass, float pE, float pPr, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int np, double3* mpos, double3* mf, double3* mm, double3& _mf, double3& _mm)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	particle_polygonObject_collision_kernel << < numBlocks, numThreads >> >(
		dpi,
		(double4 *)dsph,
		dpmi,
		E,
		pr,
		rest,
		ratio,
		fric,
		(float4 *)pos,
		(float3 *)vel,
		(float3 *)omega,
		(float3 *)force,
		(float3 *)moment,
		mass,
		pE,
		pPr,
		sorted_index,
		cstart,
		cend,
		mpos,
		mf,
		mm);
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