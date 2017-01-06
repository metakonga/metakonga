#include "cu_dem_impl.cuh"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

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

void setSymbolicParameter(device_parameters *h_paras)
{
	checkCudaErrors( cudaMemcpyToSymbol(cte, h_paras, sizeof(device_parameters)) );
}

void vv_update_position(double *pos, double *vel, double *acc, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	vv_update_position_kernel<<< numBlocks, numThreads >>>(
		(double4 *)pos,
		(double4 *)vel,
		(double4 *)acc);
}

void vv_update_velocity(double *vel, double *acc, double *omega, double *alpha, double *force, double *moment, unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	vv_update_velocity_kernel<<< numBlocks, numThreads >>>(
		(double4 *)vel,
		(double4 *)acc,
		(double4 *)omega,
		(double4 *)alpha,
		(double3 *)force,
		(double3 *)moment);
}

void cu_mergedata(double* data, double* pos, uint np, double3* vertice, uint snp)
{
	computeGridSize(np + snp, 256, numBlocks, numThreads);
	mergedata_kernel<<<numBlocks, numThreads>>>((double4*)data, (double4*)pos, np, vertice, snp);
}

void cu_calculateHashAndIndex(
	unsigned int* hash, 
	unsigned int* index, 
	double *pos, 
	unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	calculateHashAndIndex_kernel<<< numBlocks, numThreads >>>(hash, index, (double4 *)pos);

	thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),
		thrust::device_ptr<unsigned>(hash + np),
		thrust::device_ptr<unsigned>(index));
}

void cu_reorderDataAndFindCellStart(
	unsigned int* hash, 
	unsigned int* index, 
	unsigned int* cstart, 
	unsigned int* cend, 
	unsigned int* sorted_index, 
	unsigned int np, 
	unsigned int nc)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	checkCudaErrors(cudaMemset(cstart, 0xffffffff, nc*sizeof(unsigned int)) );
	unsigned smemSize=sizeof(unsigned int)*(numThreads+1);

	reorderDataAndFindCellStart_kernel<<< numBlocks, numThreads, smemSize >>>(
		hash, 
		index, 
		cstart, 
		cend, 
		sorted_index);
}

void cu_calculate_p2p(double* pos, double* vel, double* acc, double* omega, double* alpha, double* force, double* moment, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,  unsigned int np, unsigned int cRun)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	calculate_p2p_kernel<<< numBlocks, numThreads >>>(
		(double4 *)pos, 
		(double4 *)vel, 
		(double4 *)acc, 
		(double4 *)omega, 
		(double4 *)alpha, 
		(double3 *)force, 
		(double3 *)moment, 
		sorted_index, 
		cstart, 
		cend,
		cRun);
}

void cu_cube_hertzian_contact_force(
	device_plane_info *planes,
	double kn, 
	double vn, 
	double ks,
	double vs,
	double mu,
	bool* isLineContact,
	double* pos, 
	double* vel,
	double* omega,
	double* force,
	double* moment, 
	unsigned int np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	cube_hertzian_contact_force_kernel<<< numBlocks, numThreads >>>(
		planes,
		kn, 
		vn, 
		ks, 
		vs, 
		mu,
		isLineContact,
		(double4 *)pos,
		(double4 *)vel,
		(double4 *)omega,
		(double3 *)force,
		(double3 *)moment);
}

double3 cu_shape_hertzian_contact_force(
	cu_mass_info *minfo,
	cu_polygon* polygons,
	unsigned int *id_set,
	unsigned int *poly_start,
	unsigned int *poly_end, 
	double kn,
	double vn,
	double ks, 
	double vs,
	double mu,
	bool* isLineContact,
	double* pos,
	double* vel, 
	double* omega,
	double* force,
	double* moment,
	double3* bforce,
	unsigned int np,
	unsigned int* sorted_id,
	unsigned int* cell_start,
	unsigned int* cell_end)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	double3 *result_bf;
	checkCudaErrors( cudaMalloc((void**)&result_bf, sizeof(double3) * numBlocks) );
	checkCudaErrors( cudaMemset(force, 0, sizeof(double3)*np));
	checkCudaErrors( cudaMemset(moment, 0, sizeof(double3)*np));
	shpae_hertzian_contact_force_kernel<<< numBlocks, numThreads >>>(
		minfo,
		polygons,
		id_set,
		poly_start, 
		poly_end,
		kn,
		vn,
		ks,
		vs,
		mu,
		isLineContact,
		(double4*)pos,
		(double4*)vel,
		(double4*)omega,
		(double3*)force,
		(double3*)moment,
		sorted_id,
		cell_start,
		cell_end);
	
	//checkCudaErrors( cudaMemcpy(h_force, force, sizeof(double)*np*3, cudaMemcpyDeviceToHost) );
	shape_line_or_edge_contact_force_kernel<<< numBlocks, numThreads >>>(
		minfo,
		polygons,
		id_set,
		poly_start, 
		poly_end,
		kn,
		vn,
		ks,
		vs,
		mu,
		isLineContact,
		(double4*)pos,
		(double4*)vel,
		(double4*)omega,
		(double3*)force,
		(double3*)moment,
		sorted_id,
		cell_start,
		cell_end);
	//double* h_force = new double[np*3];
	//checkCudaErrors( cudaMemcpy(h_force, force, sizeof(double)*np*3, cudaMemcpyDeviceToHost) );
  	unsigned smemSize=sizeof(double3)*(256);
  	reduce6<double3, 256><<< numBlocks, numThreads, smemSize >>>((double3*)force, result_bf, np);
  	double3* h_result_bf = new double3[numBlocks];
  	double3 final_result = make_double3(0, 0, 0);
  	checkCudaErrors( cudaMemcpy(h_result_bf, result_bf, sizeof(double3) * numBlocks, cudaMemcpyDeviceToHost) );
// 	//for(unsigned int i = 0; i < np; i++){
// 		//final_result.x += h_force[i*3+0];
// 	//	final_result.y += h_force[i*3+1];
// 	//	final_result.z += h_force[i*3+2];
// 	//}
// 	//std::cout << final_result.x << " " << final_result.y << " " << final_result.z << std::endl;
	for(unsigned int i = 0; i < numBlocks; i++){
		final_result.x += h_result_bf[i].x;
		final_result.y += h_result_bf[i].y;
		final_result.z += h_result_bf[i].z;
	}
 	delete [] h_result_bf;
// 	//delete [] h_force;
 	checkCudaErrors( cudaFree(result_bf) );
 	return final_result;
}

void cu_update_polygons_position(double3* mpos, double* A, double3* local_vertice, unsigned int nvertex, double3 *pos)
{
	computeGridSize(nvertex, 256, numBlocks, numThreads);
	update_polygons_position<<< numBlocks, numThreads >>>(mpos, A, local_vertice, nvertex, pos);
}

void cu_update_polygons_information(cu_polygon* polygons, uint3 *indice, double3* pos, unsigned ni)
{
	computeGridSize(ni, 256, numBlocks, numThreads);
	update_polygons_information<<< numBlocks, numThreads >>>(polygons, (uint3 *)indice, pos, ni);
}