#include "fluid_detection.h"
#include "fluid_particle.h"
#include "sphydrodynamics.h"
#include "parSPH_V2_utils.h"
#include <thrust/sort.h>


fluid_detection::fluid_detection()
	: sph(NULL)
{

}

fluid_detection::fluid_detection(sphydrodynamics *_sph)
	: sph(_sph)
{

}

fluid_detection::~fluid_detection()
{

}

void fluid_detection::setWorldBoundary(VEC3F bMin, VEC3F bMax)
{
	gMin = bMin;
	gMax = bMax;
	gSize = bMax - bMin;
}

bool fluid_detection::initGrid()
{
	cells = 0;

	gMin = gMin - VEC3F(gcSize);
	gMax = gMax + VEC3F(gcSize);
	gSize = gMax - gMin;

	gcCount.x = static_cast<int>(ceil(gSize.x / gcSize));
	gcCount.y = static_cast<int>(ceil(gSize.y / gcSize));
	cells = gcCount.x * gcCount.y;
	if (sph->dimension() == DIM3){
		gcCount.z = static_cast<int>(ceil(gSize.z / gcSize));
		cells *= gcCount.z;
	}

	if (!gcCount.x || !gcCount.y){
		std::cout << "You need to correctly set simulation boundaries" << std::endl;
		return false;
	}

	cellCount_1 = gcCount - VEC3I(1);
	cSize_inv = 1.0f / gcSize;

	size_t np = sph->nParticle();
	hashes = new VEC2UI[np];
	cell_id = new size_t[np];		memset(cell_id, 0, sizeof(size_t)*np);
	cell_start = new size_t[cells];	memset(cell_start, 0, sizeof(size_t)*cells);

// 	if (sph->Device() == GPU){
// 		checkCudaErrors(cudaMalloc((void**)&d_hashes, sizeof(int2) * np));
// 		checkCudaErrors(cudaMalloc((void**)&d_cell_id, sizeof(uint) * np));
// 		checkCudaErrors(cudaMalloc((void**)&d_cell_start, sizeof(uint) * cells));
// 	}

	return true;
}

VEC3I fluid_detection::cellPos(VEC3F& pos)
{
	if (sph->dimension() == DIM2){
		return VEC3I(
			(int)floor((pos.x - gMin.x) * cSize_inv),
			(int)floor((pos.y - gMin.y) * cSize_inv),
			(int)0);
	}
	return VEC3I(
		(int)floor((pos.x - gMin.x) * cSize_inv),
		(int)floor((pos.y - gMin.y) * cSize_inv),
		(int)floor((pos.z - gMin.z) * cSize_inv));
}

size_t fluid_detection::cellHash(VEC3I& cell)
{
	if (sph->dimension() == DIM3){
		return cell.x + (cell.y * gcCount.x) + (cell.z * gcCount.x * gcCount.y);
	}
	return cell.y * gcCount.x + cell.x;
}

void fluid_detection::forEachSetup(fluid_particle* parI)
{
	VEC3F posI = parI->position();// : parI->auxPosition();
	cellI = cellPos(posI);
	if (sph->dimension() == DIM3){
		loopStart.x = max(cellI.x - 1, 0);
		loopStart.y = max(cellI.y - 1, 0);
		loopStart.z = max(cellI.z - 1, 0);
		loopEnd.x = min(cellI.x + 1, cellCount_1.x);
		loopEnd.y = min(cellI.y + 1, cellCount_1.y);
		loopEnd.z = min(cellI.z + 1, cellCount_1.z);
	}
	else{
		loopStart = cellPos(posI - sph->kernelSupportRadius());
		loopEnd = cellPos(posI + sph->kernelSupportRadius());
	}
	if (parI->Neighbors()->size())
		parI->Neighbors()->clear();
}

void fluid_detection::forEachNeighbor(fluid_particle* pari, VEC2UI *_hs)
{
	if (pari->particleType() == DUMMY)
		return;
	fluid_particle *ps = NULL;
	size_t hash=0;
	float QSq;
	VEC3F posDif;
	VEC2UI *hs = _hs ? _hs : hashes;
	VEC3F posi = pari->position();// : pari->auxPosition();
	VEC3F posj;
	//std::cout << pari->ID() << std::endl;
	if (sph->dimension() == DIM2){
		for (cellJ.y = loopStart.y; cellJ.y <= loopEnd.y; cellJ.y++){
			for (cellJ.x = loopStart.x; cellJ.x <= loopEnd.x; cellJ.x++){
				hash = cellHash(cellJ);
				size_t j = cell_start[hash];
				if (j != 0xffffffff){
					for (VEC2UI particleJ = hs[j]; hash == particleJ.x; particleJ = hs[++j]){
						fluid_particle *parj = sph->particle(particleJ.y);
						if (pari == parj)
							continue;
						posj = parj->position();// : parj->auxPosition();
						posDif = posi - posj;
// 						if (parj->particleType() == BOUNDARY)
// 							bool pause = true;
						QSq = posDif.dot() * sph->smoothingKernel().h_inv_sq;
						if (QSq >= sph->kernelFunction()->KernelSupprotSq())
							continue;
						fluid_particle::neighborInfo ni;
						ni.j = parj;
						ni.W = sph->kernelFunction()->sphKernel(QSq);
						ni.gradW = sph->kernelFunction()->sphKernelGrad(QSq, posDif);
// 						std::cout << cellJ.x << " " << cellJ.y << std::endl;
// 						if (pari->ID() == 2){
// 							if (cellJ.x == 10 && cellJ.y == 11)
// 								cellJ.x = 10;
// 						}
 						pari->Neighbors()->push_back(ni);
// 						if (parj->particleType() == DUMMY && sph->boundaryTreatment()==GHOST_PARTICLE_METHOD)
// 						{
// 							parj->setVelocity(pari->velocity() + (pari->Dg() / parj->Dg()) * (pari->velocity() - parj->velocity()));
// 						}
						if (pari->IsInner()){
							float dist = posDif.length();
							if (abs(dist - sph->particleSpacing()) < 1e-9f){
								pari->NeighborsInner().push_back(parj->ID());
							}
						}
					}
				}
			}
		}
	}
	else{
		for (cellJ.x = loopStart.x; cellJ.x <= loopEnd.x; cellJ.x++){
			for (cellJ.y = loopStart.y; cellJ.y <= loopEnd.y; cellJ.y++){
				for (cellJ.z = loopStart.z; cellJ.z <= loopEnd.z; cellJ.z++){
					hash = cellHash(cellJ);
					size_t j = cell_start[hash];
					if (j != 0xffffffff){
						/*end_index = cell_end[hash];*/
						for (VEC2UI particleJ = hs[j]; hash == particleJ.x; particleJ = hs[++j]){
							fluid_particle *parj = sph->particle(particleJ.y);
							if (pari == parj)
								continue;

							posDif = pari->auxPosition() - parj->auxPosition();
							QSq = posDif.dot() * sph->smoothingKernel().h_inv_sq;
							if (QSq >= sph->kernelFunction()->KernelSupprotSq())
								continue;
							fluid_particle::neighborInfo ni;
							ni.j = parj;
							ni.gradW = sph->kernelFunction()->sphKernelGrad(QSq, posDif);
							pari->Neighbors()->push_back(ni);
						}
					}
				}
			}
		}
	}
	switch (sph->correction()){
	case GRADIENT_CORRECTION:
		//sph->invCorrectionGradient(pari->ID());
		break;
	}

// 	if (sph->boundaryTreatment() == GHOST_PARTICLE_METHOD)
// 	{
// 		sph->setGhostParticles(pari);
// 	}
}

void fluid_detection::resizeDetectionData(size_t pnp, size_t numg)
{
	VEC2UI *_hs = new VEC2UI[pnp + numg];
	size_t *_ci = new size_t[pnp + numg];
	memcpy(_hs, hashes, sizeof(VEC2UI) * pnp);
	memcpy(_ci, cell_id, sizeof(size_t) * pnp);
	delete[] hashes;
	delete[] cell_id;
	hashes = new VEC2UI[pnp + numg];
	cell_id = new size_t[pnp + numg];
}

void fluid_detection::sort(bool isf)
{
	_isf = isf;
	if(sph->boundaryTreatment() == GHOST_PARTICLE_METHOD)
		sph->initializeGhostMap();
	fluid_particle *parI;
	size_t rnp = 0;
	VEC3F pos;
	for (size_t i = 0; i < sph->nRealParticle(); i++){

		parI = sph->particle(i);
		pos = parI->position();// : parI->auxPosition();
		hashes[i] = VEC2UI(cellHash(cellPos(pos)), i);
		cell_id[i] = hashes[i].x;
	}
	memset(cell_start, 0xffffffff, sizeof(size_t)*cells);
	thrust::sort_by_key(cell_id, cell_id + sph->nRealParticle(), hashes);

	size_t hash_start = hashes[0].x;
	cell_start[hash_start] = 0;
	for (size_t i = 1; i < sph->nRealParticle(); i++){
		if (hash_start != hashes[i].x){
			hash_start = hashes[i].x;
			if (hash_start > cells){
				VEC3F p = sph->particle(hashes[i].y)->position();
				std::cout << ".....error : hash_start is " << hash_start << std::endl;
				std::cout << ".....error position : [ " << p.x << ", " << p.y << ", " << p.z << " ]" << std::endl;
			}
			cell_start[hash_start] = i;
		}
		
	}
// 	if (sph->correction() == GRADIENT_CORRECTION){
// 		sph->clearMatKgc();
// 	}
	if (sph->boundaryTreatment() == GHOST_PARTICLE_METHOD){
		size_t nGhost = 0;
		for (size_t i = sph->nParticleByType(FLUID); i < sph->nRealParticle(); i++){
			forEachSetup(sph->particle(i));
			nGhost += createGhostParticles(i, true);
		}
		//resizeDetectionData(sph->nRealParticle(), nGhost);
		sph->resizeParticle(nGhost);
		sph->transferGhostParticle();
	//	sph->exportParticlePosition();
		VEC2UI *_hs = new VEC2UI[sph->nRealParticle() + nGhost];
		size_t *_ci = new size_t[sph->nRealParticle() + nGhost];
		for (size_t i = 0; i < sph->nParticle(); i++){	
			parI = sph->particle(i);
			_hs[i] = VEC2UI(cellHash(cellPos(parI->auxPosition())), i);
			_ci[i] = _hs[i].x;
		}
		memset(cell_start, 0xffffffff, sizeof(size_t)*cells);
		thrust::sort_by_key(_ci, _ci + sph->nParticle(), _hs);

		size_t hash_start = hashes[0].x;
		cell_start[hash_start] = 0;
		for (size_t i = 1; i < sph->nParticle(); i++){
			if (hash_start != _hs[i].x){
				hash_start = _hs[i].x;
				if (hash_start > cells){
					VEC3F p = sph->particle(_hs[i].y)->position();
					std::cout << ".....error : hash_start is " << hash_start << std::endl;
					std::cout << ".....error position : [ " << p.x << ", " << p.y << ", " << p.z << " ]" << std::endl;
				}
				cell_start[hash_start] = i;
			}

		}
		for (size_t i = 0; i < sph->nParticle(); i++){
// 			if (i == 38)
// 				i = 38;
			forEachSetup(sph->particle(i));
			forEachNeighbor(sph->particle(i), _hs);
		}
		return;
	}
	for (size_t i = 0; i < sph->nParticle(); i++){
		forEachSetup(sph->particle(i));
		forEachNeighbor(sph->particle(i));
	}
}

size_t fluid_detection::createGhostParticles(size_t i, bool isOnlyCount)
{
	size_t hash;
	VEC3F posDif;
	VEC3F mp; // middle point
	float proj_d;
	float df = 0.f;
	float dg = 0.f;
	float r = 0.f;
	size_t hash2;
	size_t count = 0;
	if (i == 7921)
		i = 7921;
	fluid_particle *pari = sph->particle(i);
	if (sph->dimension() == DIM2){
		for (cellJ.y = loopStart.y; cellJ.y <= loopEnd.y; cellJ.y++){
			for (cellJ.x = loopStart.x; cellJ.x <= loopEnd.x; cellJ.x++){
				hash = cellHash(cellJ);
				size_t j = cell_start[hash];
				if (j != 0xffffffff){
					/*end_index = cell_end[hash];*/
					for (VEC2UI particleJ = hashes[j]; hash == particleJ.x; particleJ = hashes[++j]){
						fluid_particle *parj = sph->particle(particleJ.y);
						if (pari == parj)
							continue;
						
						if (parj->particleType() == FLUID){
							VEC3F fpos = _isf ? parj->position() : parj->auxPosition();
							VEC3F fvel = _isf ? parj->velocity() : parj->auxVelocity();
							fluid_particle ng;
							ng.setType(DUMMY);
// 							if (parj->ID() == 0)
// 								bool pause = true;
							posDif = pari->position() - fpos;
							float QSq = posDif.dot() * sph->smoothingKernel().h_inv_sq;
							if (QSq >= sph->kernelFunction()->KernelSupprotSq())
								continue;
							VEC3F lp1 = pari->position() - pari->tangent();
							VEC3F lp2 = pari->position() + pari->tangent();
							ng.setPosition(utils::calcMirrorPosition2Line(lp1, lp2, fpos, mp));
							ng.setAuxPosition(ng.position());
							mp = VEC3F(0.5f*(fpos.x + ng.position().x), 0.5f * (fpos.y + ng.position().y), 0.f);
							df = (fpos - mp).length();
							dg = (ng.position() - mp).length();
							r = fpos.y - ng.position().y;//(ng.auxPosition() - mp).dot(VEC3F(0.f, -1.f, 0.f));
							ng.setPressure(parj->pressure() + sph->density()*sph->gravity().length()*r);
							//ng.setDg(dg);
							ng.setAddGhostPressure(sph->density()*sph->gravity().length()*r);
							ng.setBaseFluid(parj->ID());
							ng.setMass(parj->mass());
							//ng.setDf(df);
							//parj->setDistanceFromTangentialBoundary(df);
							ng.setVelocity(pari->velocity() + (df / dg) * (pari->velocity() - fvel));
							ng.setAuxVelocity(ng.velocity());
							ng.setDensity(1000.f);
							//ng.setDistanceGhost((ng.position() - mp).length());
							hash2 = cellHash(cellPos(ng.position()));
							if (sph->insertGhostParticle(hash2, ng)){
								count++;
							}
						}
					}
				}
			}
		}
	}
	return count;
}