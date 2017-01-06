#include "s_grid.h"
#include "sphydrodynamics.h"
#include "cu_sph_decl.cuh"
#include <cuda_runtime.h>


#include <thrust/sort.h>
using namespace parsph;

grid::grid(sphydrodynamics *_sph)
	: sph(_sph)
	, hashes(NULL)
	, cell_id(NULL)
	, cell_start(NULL)
	// initialize device memory
	, d_hashes(NULL)
	, d_cell_id(NULL)
	, d_cell_start(NULL)
{

}

grid::~grid() 
{
	if(hashes) delete [] hashes; hashes = NULL;
	if(cell_id) delete [] cell_id; cell_id = NULL;
	if(cell_start) delete [] cell_start; cell_start = NULL;

	// dealloc device memory
	if(d_hashes) cudaFree(d_hashes); d_hashes = NULL;
	if(d_cell_id) cudaFree(d_cell_id); d_cell_id = NULL;
	if(d_cell_start) cudaFree(d_cell_start); d_cell_start = NULL;
}

bool grid::initGrid()
{
	cells = 0;

	gridMin = sph->gridMin - vector3<double>(sph->gridCellSize);
	gridMax = sph->gridMax + vector3<double>(sph->gridCellSize);
	gridSize = gridMax - gridMin;
	gridCellSize = sph->gridCellSize;

	gridCellCount.x = static_cast<int>(ceil(gridSize.x / gridCellSize));
	gridCellCount.y = static_cast<int>(ceil(gridSize.y / gridCellSize));
	cells = gridCellCount.x * gridCellCount.y;
	if(sph->dimension == DIM3){
		gridCellCount.z = static_cast<int>(ceil(gridSize.z / gridCellSize));
		cells *= gridCellCount.z;
	}

	if(!gridCellCount.x || !gridCellCount.y){
		std::cout << "You need to correctly set simulation boundaries" << std::endl;
		return false;
	}

	cellCount_1 = gridCellCount - vector3<int>(1);
	cellSizeInv = 1.0 / gridCellSize;

	unsigned int np = sph->ParticleCount();
	hashes = new vector2<int>[np];
	cell_id = new unsigned int[np];		memset(cell_id, 0, sizeof(unsigned int)*np);
	cell_start = new unsigned int[cells];	memset(cell_start, 0, sizeof(unsigned int)*cells);

	if(sph->Device() == GPU){
		checkCudaErrors( cudaMalloc((void**)&d_hashes, sizeof(int2) * np) );
		checkCudaErrors( cudaMalloc((void**)&d_cell_id, sizeof(uint) * np) );
		checkCudaErrors( cudaMalloc((void**)&d_cell_start, sizeof(uint) * cells) );
	}

 	return true;
}

vector3<int> grid::CellPos(vector3<double> pos)
{
	if(sph->Dimension()==DIM2){
		return vector3<int>(
			(int)floor((pos.x - gridMin.x) * cellSizeInv),
			(int)floor((pos.y - gridMin.y) * cellSizeInv),
			(int)0.0);
	}
	return vector3<int>(
		(int)floor((pos.x - gridMin.x) * cellSizeInv), 
		(int)floor((pos.y - gridMin.y) * cellSizeInv),
		(int)floor((pos.z - gridMin.z) * cellSizeInv));
}

unsigned int grid::CellHash(vector3<int> cell)
{
	if(sph->Dimension() == DIM3){
		return cell.x + (cell.y * gridCellCount.x) + (cell.z * gridCellCount.x * gridCellCount.y);
	}
	return cell.y * gridCellCount.x + cell.x;
}



void grid::reorderDataAndFindCellStart(size_t ID, size_t begin, size_t end)
{

}

void grid::findOutOfBound()
{

}

void grid::sort()
{
	s_particle *parI;
	unsigned int np = sph->ParticleCount();
	// Hash value calculation
	for(unsigned int i = 0; i < np; i++){
		if(i == 1912)
		{
			bool pause = true;
		}
		parI = sph->getParticle(i);
		hashes[i] = vector2<int>(CellHash(CellPos(parI->Position())), i);
// 		if(hashes[i].x > cells)
// 		{
// 			bool apucs = true;
// 		}
		cell_id[i] = hashes[i].x;
	}
	memset(cell_start, 0xffffffff, sizeof(unsigned int)*cells);
	thrust::sort_by_key(cell_id, cell_id + np, hashes);

// 	std::fstream pf;
// 	pf.open("C:/C++/h_hashes.txt", std::ios::out);
// 	for(unsigned int i = 0; i < sph->ParticleCount(); i++){
// 		pf << hashes[i].x << " " << hashes[i].y << std::endl;
// 	}
// 	pf.close();

	unsigned int hash_start = hashes[0].x;
	cell_start[hash_start] = 0;
	for(unsigned int i = 1; i < np; i++){
		if(hash_start != hashes[i].x){
			hash_start = hashes[i].x;		
			if(hash_start > cells){
				vector3<double> p = sph->getParticle(hashes[i].y)->Position();
				std::cout << ".....error : hash_start is " << hash_start << std::endl;
				std::cout << ".....error position : [ " << p.x << ", " << p.y << ", " << p.z << " ]" << std::endl;
			}
			cell_start[hash_start] = i;
		}
	}
// 	pf.open("C:/C++/h_cs.txt", std::ios::out);
// 	for(unsigned int i = 0; i < cells; i++){
// 		pf << i << " " << cell_start[i] << std::endl;
// 	}
// 	pf.close();
// 	std::fstream pf;
// 	pf.open("C:/C++/h_hashes.txt", std::ios::out);
// 	for(unsigned int i = 0; i < sph->ParticleCount(); i++){
// 		pf << hashes[i].x << " " << hashes[i].y << std::endl;
// 	}
// 	pf.close();
	if(sph->Correction() == GRADIENT_CORRECTION){
		sph->ClearMatKgc();
	}
	for(unsigned int i = 0; i < np; i++){
		forEachSetup(sph->getParticle(i));
		forEachNeighbor(sph->getParticle(i));
	}
}

void grid::forEachSetup(s_particle* parI)
{
	vector3<double> posI = parI->Position();
	cellI = CellPos(posI);
	if(sph->Dimension() == DIM3){
		loopStart.x = max(cellI.x - 1, 0);
		loopStart.y = max(cellI.y - 1, 0);
		loopStart.z = max(cellI.z - 1, 0);
		loopEnd.x = min(cellI.x + 1, cellCount_1.x);
		loopEnd.y = min(cellI.y + 1, cellCount_1.y);
		loopEnd.z = min(cellI.z + 1, cellCount_1.z);
	}
	else{
		loopStart = CellPos(posI - sph->kernelSupportRadius);
		loopEnd = CellPos(posI + sph->kernelSupportRadius);
	}
	
	parI->Neighbors().clear();
}

void grid::forEachNeighbor(s_particle* pari)
{
	s_particle *ps = NULL;
	int hash;
	double QSq;
// 	if(pari->ID() == 291){
// 		bool pause = true;
// 	}
	vector3<double> posDif;
	//unsigned int start_index, end_index;
	if(sph->Dimension() == DIM2){
		for(cellJ.x = loopStart.x; cellJ.x <= loopEnd.x; cellJ.x++){
			for(cellJ.y = loopStart.y; cellJ.y <= loopEnd.y; cellJ.y++){
				hash = CellHash(cellJ);
				unsigned int j = cell_start[hash];
				if(j != 0xffffffff){
					/*end_index = cell_end[hash];*/
					for(vector2<int> particleJ = hashes[j]; hash ==particleJ.x; particleJ = hashes[++j]){
						s_particle *parj = sph->getParticle(particleJ.y);
						if(pari == parj)
							continue;
						
						posDif = pari->Position() - parj->Position();
						QSq = posDif.dot() * sph->skernel.h_inv_sq;
						if(QSq >= sph->sphkernel->KernelSupprotSq()) 
							continue;
						s_particle::neighbor_info ni;
						ni.j = parj;
						ni.W = sph->sphkernel->sphKernel(QSq);
						ni.gradW = sph->sphkernel->sphKernelGrad(QSq, posDif);
						switch(sph->Correction()){
						case GRADIENT_CORRECTION:
							sph->CorrectionGradient(pari->ID(), parj, ni.gradW, -posDif);
						}
						pari->Neighbors().push_back(ni);

						if(pari->IsInner()){
							double dist = posDif.length();
							if(abs(dist - sph->particleSpacing()) < 1e-9){
								pari->NeighborsInner().push_back(parj->ID());
							}
						}
					}
				}
			}
		}
	}
	else{
		for(cellJ.x = loopStart.x; cellJ.x <= loopEnd.x; cellJ.x++){
			for(cellJ.y = loopStart.y; cellJ.y <= loopEnd.y; cellJ.y++){
				for(cellJ.z = loopStart.z; cellJ.z <= loopEnd.z; cellJ.z++){
					hash = CellHash(cellJ);
					unsigned int j = cell_start[hash];
					if(j != 0xffffffff){
						/*end_index = cell_end[hash];*/
						for(vector2<int> particleJ = hashes[j]; hash ==particleJ.x; particleJ = hashes[++j]){
							s_particle *parj = sph->getParticle(particleJ.y);
							if(pari == parj)
								continue;
							
							posDif = pari->Position() - parj->Position();
							QSq = posDif.dot() * sph->skernel.h_inv_sq;
							if(QSq >= sph->sphkernel->KernelSupprotSq()) 
								continue;
							s_particle::neighbor_info ni;
							ni.j = parj;
							ni.gradW = sph->sphkernel->sphKernelGrad(QSq, posDif);
							pari->Neighbors().push_back(ni);
						}
					}
				}
			}
		}
	}
	switch(sph->Correction()){
	case GRADIENT_CORRECTION:
		sph->invCorrectionGradient(pari->ID());
	}
}

void grid::cusort()
{
	// Hash value calculation
	cu_calcHashValue(d_hashes, d_cell_id, sph->d_auxPos, sph->ParticleCount());
// 	cudaMemcpy(hashes, d_hashes, sizeof(int) * 2 * sph->ParticleCount(), cudaMemcpyDeviceToHost);
// 	std::fstream pf;
// 	pf.open("C:/C++/d_hashes.txt", std::ios::out);
// 	for(unsigned int i = 0; i < sph->ParticleCount(); i++){
// 		pf << hashes[i].x << " " << hashes[i].y << std::endl;
// 	}
// 	pf.close();
	cu_reorderDataAndFindCellStart(d_hashes, d_cell_start, sph->ParticleCount(), cells);
// 	cudaMemcpy(cell_start, d_cell_start, sizeof(uint) * cells, cudaMemcpyDeviceToHost);
// 
// 	pf.open("C:/C++/d_cs.txt", std::ios::out);
// 	for(unsigned int i = 0; i < cells; i++){
// 		pf << i << " " << cell_start[i] << std::endl;
// 	}
// 	pf.close();
//	checkCudaErrors( cudaMemcpy(hashes, d_hashes, sizeof(int2) * sph->ParticleCount(), cudaMemcpyDeviceToHost) );

// 	std::fstream pf;
// 	pf.open("C:/C++/d_hashes.txt", std::ios::out);
// 	for(unsigned int i = 0; i < sph->ParticleCount(); i++){
// 		pf << hashes[i].x << " " << hashes[i].y << std::endl;
// 	}
// 	pf.close();
}

