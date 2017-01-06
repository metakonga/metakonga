#include "cu_sph_impl.cuh"
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

void cu_calcHashValue(int2 *hashes, uint *cell_id, double3 *pos, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	calculateHashAndIndex_kernel<<< numBlocks, numThreads >>>(hashes, cell_id, pos);

	thrust::sort_by_key(thrust::device_ptr<unsigned>(cell_id),
		thrust::device_ptr<unsigned>(cell_id + np),
		thrust::device_ptr<int2>(hashes) );
}

void cu_reorderDataAndFindCellStart(int2 *hashes, uint* cell_start, uint np, uint nc)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	checkCudaErrors( cudaMemset(cell_start, 0xffffffff, nc*sizeof(uint)) );
	unsigned smemSize=sizeof(unsigned int)*(numThreads+1);

	reorderDataAndFindCellStart_kernel<<< numBlocks, numThreads, smemSize >>>(hashes, cell_start);
}

void cu_auxiliaryPosition(t_particle* pclass, double3 *pos, double3 *vel, double3* tpos, bool* isFloating, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	auxiliaryPosition_kernel<<<numBlocks, numThreads>>>(pclass, pos, vel, tpos, isFloating);
}
void cu_auxiliaryVelocity(t_particle* type, int2 *hashes, uint* cell_start, double3 *pos, double3 *vel, double3 *auxVel, bool* isFloating, double* eddyVisc, double6 *matKgc, double3* gamma, double *sumKernel, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	auxiliaryVelocity_kernel<<< numBlocks, numThreads >>>(type, hashes, cell_start, pos, vel, auxVel, isFloating, eddyVisc, matKgc, gamma, sumKernel);
}

void cu_calcFreeSurface(t_particle* pclass, int2 *hashes, uint* cell_start, double3 *pos, bool* freesurface, double* divP, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	freeSurface_kernel<<< numBlocks, numThreads >>>(pclass, hashes, cell_start, pos, freesurface, divP);
}

void cu_predictor(t_particle* pclass, bool* fsurface, int2 *hashes, uint* cell_start, double3 *pos, double3* vel, double *rhs, bool* isFloating, double6 *matKgc, double3* gamma, double *sumKernel, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	predictionStep_kernel<<< numBlocks, numThreads >>>(pclass, fsurface, hashes, cell_start, pos, vel, rhs, isFloating, matKgc, gamma, sumKernel);
}

void cu_PPEquation(t_particle* pclass, int2 *hashes, uint* cell_start, double3* pos, double3* vel, double* press, double* out, double6 *matKgc, double* hpressure, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	PPE_kernel<<<numBlocks, numThreads >>>(pclass, hashes, cell_start, pos, vel, press, out, matKgc, hpressure);
}

void cu_PPEquation_PPESolver(t_particle* pclass, int2 *hashes, uint* cell_start, double3* pos, double* press, double* out, double6 *matKgc, double3* gamma, double *sumKernel, bool* freesurface, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	PPE_kernel_PPESolver<<<numBlocks, numThreads >>>(pclass, hashes, cell_start, pos, press, out, matKgc, gamma, sumKernel, freesurface);
}

double initPPESolver(t_particle* pclass, double* rhs, double* lhs, double* residual, double* conjugate0, double* conjugate1, double* tmp0, double* tmp1, bool* freesurface, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	
	initPPE_kernel<<< numBlocks, numThreads >>>(pclass, rhs, lhs, residual, conjugate0, conjugate1, tmp0, tmp1, freesurface);
// 	double* h_re = new double[np];
// 	cudaMemcpy(h_re, residual, sizeof(double)*np, cudaMemcpyDeviceToHost);
// 	double sum = 0;
// 	for(unsigned int i = 0; i < np; i++){
// 		sum+=h_re[i] * h_re[i];
// 	}
// 	double sum2 = dot6(pclass, freesurface, residual, residual, np);
// 	delete [] h_re;
	return dot6(pclass, freesurface, residual, residual, np);
	
}

double dot6(t_particle *type, bool* freesurface, double* d1, double *d2, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	uint smemSize = sizeof(double) * numThreads;
	double *out;
	checkCudaErrors( cudaMalloc((void**)&out, sizeof(double) * numBlocks) );
	dot6_kernel<double, 256><<< numBlocks, numThreads, smemSize >>>(type, freesurface, d1, d2, out);
	double *h_out = new double[numBlocks];
	checkCudaErrors( cudaMemcpy(h_out, out, sizeof(double)*numBlocks, cudaMemcpyDeviceToHost) );
	double sum = 0;
	for(unsigned int i = 0; i < numBlocks; i++){
		sum += h_out[i];
	}
	delete [] h_out;
	checkCudaErrors( cudaFree(out) );
	return sum;
}

void cu_dummyScalarCopy(t_particle* pclass, double *vec, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	dummyScalarCopy_kernel<<< numBlocks, numThreads>>>(pclass, vec);
}

void cu_updatePressureAndResidual(double alpha, double* conjugate0, double omega, double* conjugate1, double* tmp1, double* pressure, double* residual, bool* freesurface, t_particle *type, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	updatePressureAndResidual_kernel<<<numBlocks, numThreads>>>(alpha, conjugate0, omega, conjugate1, tmp1, pressure, residual, freesurface, type);
}

void cu_updateConjugate(double* conjugate0, double* residual, double* tmp0, double beta, double omega, bool* freesurface, t_particle* type, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	updateConjugate_kernel<<< numBlocks, numThreads >>>(conjugate0, residual, tmp0, beta, omega, freesurface, type);
}

void cu_setPressureFreesurfaceAndDummyParticle(t_particle *pclass, bool* free_surface, double* pressure, double* hpressure, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	setPressureFreesurfaceAndDummyParticle_kernel<<< numBlocks, numThreads >>>(pclass, free_surface, pressure, hpressure);
}

void cu_corrector(t_particle* pclass, int2 *hashes, uint* cell_start, double3 *pos, double3* auxPos, double3 *vel, double3 *auxVel, double3* gradP,  double* pressure, bool* isFloating, double6 *matKgc, double3 *gamma, double *sumKernel, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	corrector_kernel<<< numBlocks, numThreads >>>(pclass, hashes, cell_start, pos, auxPos, vel, auxVel, gradP, pressure, isFloating, matKgc, gamma, sumKernel);
}

void cu_shifting(t_particle* pclass, int2 *hashes, uint* cell_start, bool* freesurface, double3 *tpos, double3 *tvel, double* tpressure, double3* pos, double3* vel, double* pressure, double6 *matKgc, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	shiftingParticles_kernel<<<numBlocks, numThreads>>>(pclass, hashes, cell_start, freesurface, tpos, tvel, pos);
	shiftingUpdate<<< numBlocks, numThreads >>>(pclass, hashes, cell_start, freesurface, tpos, tvel, tpressure, pos, vel, pressure, matKgc);
}

void cu_calcGradientCorrection(t_particle* pclass, int2 *hashes, uint* cell_start, double3 *pos, double6 *matKgc, double3 *gamma, double *sumKernel, uint np, double *density)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	gradientCorrection_kernel<<< numBlocks, numThreads>>>(pclass, hashes, cell_start, pos, matKgc, gamma, sumKernel, density);
}

void cu_runExpression(double3* pos, double3* vel, double time, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	runExpression_kernel<<< numBlocks, numThreads >>>(pos, vel, time, np);
}

void cu_setInnerParticlePressureForDummyParticle(t_particle* pclass, int2 *hashes, uint* cell_start, double3 *pos, double* pressure, bool* isInner, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	setInnerParticlePressureForDummyParticle<<< numBlocks, numThreads >>>(pclass, hashes, cell_start, pos, pressure, isInner);
}

void cu_findLineSegmentIndex(t_particle* type, int2* hashes, uint *cell_start, double3* pos, double3* gradP, double* pressure, bool* isFloating, double3* Pf, double3* sp, double3* ep, double3* n, uint* seg_n, uint particleCount)
{
	computeGridSize(particleCount, 256, numBlocks, numThreads);
	findLineSegmentIndex_kernel<<< numBlocks, numThreads>>>(type, hashes, cell_start, pos, gradP, pressure, isFloating, Pf, sp, ep, n, seg_n);
}

void cu_updateBodyInformation(t_particle* dclass, double3 *pos, double3 *vel, double3 bforce, bool* floatingBodyParticle, double3* sp, double3* ep, uint particleCount)
{
	computeGridSize(particleCount, 256, numBlocks, numThreads);
	double3 *d_bforce;
	cudaMalloc((void**)&d_bforce, sizeof(double3));
	cudaMemcpy(d_bforce, &bforce, sizeof(double3), cudaMemcpyHostToDevice);
	updateBodyInformation_kernel<<< numBlocks, numThreads>>>(dclass, pos, vel, d_bforce, floatingBodyParticle, sp, ep);
	cudaFree(d_bforce);
}

void cu_calcEddyViscosity(t_particle* pclass, int2 *hashes, uint* cell_start, double3* pos, double3* vel, double* density, double *eddyVisc, uint np)
{
	computeGridSize(np, 256, numBlocks, numThreads);
	calcEddyViscosity_kernel<<< numBlocks, numThreads >>>(pclass, hashes, cell_start, pos, vel, density, eddyVisc);
}