#include "cu_particleSystem.cuh"
#include "cu_particleSystem_impl.cuh"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <iostream>

unsigned numThreads, numBlocks;

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

void setSymbolicParameter(parameters *h_paras)
{
	checkCudaErrors( cudaMemcpyToSymbol(paras, h_paras, sizeof(parameters)) );
}

void setEulerParameters(float *eup, float *euv, float *eua, float *omega, /*float *gomega,*/ float *alpha, unsigned nP)
{
	computeGridSize(nP, 256, numBlocks, numThreads);
	setEulerParametersKernel<<< numBlocks, numThreads >>>(
		(float4 *)eup, 
		(float4 *)euv, 
		(float4 *)eua, 
		(float3 *)omega,
		//(float3 *)gomega,
		(float3 *)alpha);
}

void updatePosition(float *pos, float *vel, float *acc, float *eup, float *euv, float *eua, unsigned nP)
{
	computeGridSize(nP, 256, numBlocks, numThreads);
	updatePositionKernel<<< numBlocks, numThreads >>>(
		(float3 *)pos,
		(float3 *)vel,
		(float3 *)acc,
		(float4 *)eup,
		(float4 *)euv,
		(float4 *)eua);
}

void calculateHashAndIndex(unsigned *hash, unsigned *index, float *pos, unsigned nP)
{
	computeGridSize(nP, 256, numBlocks, numThreads);
	calculateHashAndIndexKernel<<< numBlocks, numThreads >>>(
		hash,
		index,
		(float3 *)pos);
	unsigned* h_gridParticleHash = new unsigned[nP];
	unsigned* h_gridParticleIndex = new unsigned[nP];
	//checkCudaErrors( cudaMemcpy(h_gridParticleHash, hash, sizeof(unsigned)*nP, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy(h_gridParticleIndex, index, sizeof(unsigned)*nP, cudaMemcpyDeviceToHost) );
	//std::cout << h_gridParticleHash+nP << std::endl;
	thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),
		thrust::device_ptr<unsigned>(hash + nP),
		thrust::device_ptr<unsigned>(index));
	 checkCudaErrors( cudaMemcpy(h_gridParticleHash, hash, sizeof(unsigned)*nP, cudaMemcpyDeviceToHost) );
	 checkCudaErrors( cudaMemcpy(h_gridParticleIndex, index, sizeof(unsigned)*nP, cudaMemcpyDeviceToHost) );
		delete [] h_gridParticleIndex;
	 	delete [] h_gridParticleHash;
}

void reorderDataAndFindCellStart(
	float *sortedPos,
	float *sortedVel,
	float *sortedOmega,
	float *pos,
	float *vel,
	float *omega,
	unsigned *cellStart, 
	unsigned *cellEnd, 
	unsigned *gridParticleHash, 
	unsigned *gridParticleIndex, 
	unsigned nP, 
	unsigned nC)
{
	computeGridSize(nP, 256, numBlocks, numThreads);
	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, nC*sizeof(unsigned)) );

	unsigned smemSize=sizeof(unsigned)*(numThreads+1);
	reorderDataAndFindCellStartKernel<<< numBlocks, numThreads, smemSize >>>(
		cellStart,
		cellEnd,
		(float3 *) sortedPos,
		(float3 *) sortedVel,
		(float3 *) sortedOmega,
		gridParticleHash,
		gridParticleIndex,
		(float3 *) pos,
		(float3 *) vel,
		(float3 *) omega);
}

void calculateCollideForce(
	float *vel, 
	float *omega, 
	float *sortedPos,
	float *sortedVel,
	float *sortedOmega,
	boundaryType *boundary,
	float *force,
	float *moment,
	unsigned *particleIndex,
	unsigned *CellStart,
	unsigned *CellEnd,
	unsigned nP,
	unsigned nC)
{
	computeGridSize(nP, 64, numBlocks, numThreads);
	calculateCollideForceKernel<<< numBlocks, numThreads >>>(
		(float3 *)vel,
		(float3 *)omega,
		(float3 *)sortedPos,
		(float3 *)sortedVel,
		(float3 *)sortedOmega,
		(float3 *)force,
		(float3 *)moment,
		boundary,
		particleIndex,
		CellStart,
		CellEnd);
}

void updateVelocity(float *vel, float *acc, float *omega, float *alpha, float *force, float *moment, unsigned nP)
{
	computeGridSize(nP, 256, numBlocks, numThreads);
	updateVelocityKernel<<< numBlocks, numThreads >>>(
		(float3 *)vel, 
		(float3 *)acc,
		(float3 *)omega, 
		(float3 *)alpha,
		(float3 *)force, 
		(float3 *)moment);
}