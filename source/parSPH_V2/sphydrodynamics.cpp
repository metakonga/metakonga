#include "sphydrodynamics.h"
#include "fluid_detection.h"

#include <stdio.h>
#include <windows.h>
#include <shlwapi.h>
#include <direct.h>
#include <stack>
#include <queue>
#include <set>
#pragma comment(lib, "shlwapi.lib")

sphydrodynamics::sphydrodynamics()
	: fp(NULL)
	, fd(NULL)
	, sphkernel(NULL)
	, InnerCornerDummyPressureIndex(NULL)
	, matKgc(NULL)
	, tCorr(CORRECTION)
	, nghost(0)
	, maxVel(0)
	, maxAcc(0)
{
	numInnerCornerParticles = 0;
	
}

sphydrodynamics::sphydrodynamics(std::string _path, std::string _name)
	: name(_name)
	, path(_path)
	, fp(NULL)
	, fd(NULL)
	, sphkernel(NULL)
	, InnerCornerDummyPressureIndex(NULL)
	, matKgc(NULL)
	, tCorr(CORRECTION)
	, nghost(0)
	, maxVel(0)
	, maxAcc(0)
{
	numInnerCornerParticles = 0;
	std::string cpath = _path + _name;
	const char* _p = cpath.c_str();
	wchar_t uni[255];
	int n = MultiByteToWideChar(CP_ACP, 0, _p, (int)strlen(_p), uni, sizeof(uni));
	uni[n] = 0;
	if (!PathIsDirectory(uni))
	{
		_mkdir(cpath.c_str());
		std::cout << "Directory path was created. - " << cpath << " -" << std::endl;
	}
	if (!fs.is_open())
	{
		fs.open(cpath + "/" + _name + ".sph", std::ios::out);
		fs << "model_path " << cpath << std::endl;
		fs << "model_name " << _name << std::endl;
	}
}

sphydrodynamics::~sphydrodynamics()
{
	if (fp) delete [] fp; fp = NULL;
	if (fd) delete fd; fd = NULL;
	if (matKgc) delete [] matKgc; matKgc = NULL;
	if (sphkernel) delete sphkernel; sphkernel = NULL;
	if (InnerCornerDummyPressureIndex) delete[] InnerCornerDummyPressureIndex; InnerCornerDummyPressureIndex = NULL;
	//if (eData) delete[] eData; eData = NULL;
	////if (pressure) delete[] pressure; pressure = NULL;
}

fluid_detection* sphydrodynamics::createFluidDetection()
{
	if (!fd)
		fd = new fluid_detection(this);
	return fd;
}

bool sphydrodynamics::isCornerOverlapping(const VEC3F& position)
{
	for (size_t i = 0; i < overlappingCorners.size(); i++){
		if ((overlappingCorners[i].c1.position - position).length() < 1e-9)
			return true;
	}
	return false;
}

size_t sphydrodynamics::initDummies(size_t wallId, VEC3F& pos, VEC3F& normal, bool onlyCountParticles, bool considerHp, int minusCount, bool isf)
{
	size_t layers = (size_t)(fd->gridCellSize() / pspace)+1;
	layers = layers - size_t(minusCount) - 1;
	if (!onlyCountParticles){
		for (size_t i = 1; i <= layers; i++){
			float hp = considerHp ? rho * -9.80665f * i * (-pspace) : 0.0f;
// 			if (considerHp)
// 			{
// 				bool pause = true;
// 			}
			initDummyParticle(wallId + i, pos - (i*pspace) * normal, hp, false, isf, false);
		}
	}
	return layers;
}

void sphydrodynamics::initDummyParticle(size_t id, VEC3F& position, float hydrop, bool isInner, bool isf, bool isMov)
{
	fluid_particle* p = &fp[id];
	p->setID(id);
	p->setType(DUMMY);
	p->setPosition(position);
	p->setDensity(rho);
	p->setIsFloating(isf);
	p->setMass(particleMass[DUMMY]);
	p->setPressure(0.f);
	p->setHydroPressure(hydrop);
	p->setVelocity(VEC3F(0.0f, 0.0f, 0.0f));
	p->setMovement(isMov);
	p->setIsInner(isInner);
	particleCountByType[DUMMY] += 1;
}

void sphydrodynamics::clearMatKgc()
{
	memset(matKgc, 0, sizeof(float6) * np);
// 	const float6 zero_mat = { 0, 0, 0, 0, 0, 0 };
// 	for (size_t i = 0; i < np; i++){
// 		//if (fp[i].particleType() == FLUID)
// 		matKgc[i] = zero_mat;
// 	}
}

bool sphydrodynamics::particleCollision(VEC3F& position, bool isfirst)
{
	if (position.x > fd->gridMax().x || position.y > fd->gridMax().y)
		return true;
	if (position.x < fd->gridMin().x || position.y < fd->gridMin().y)
		return true;

	float radius = 0.f;// pspace * 0.255f;
	//if (isfirst)
		//radius = pspace * 0.255f;
	for (std::multimap<std::string, geo::geometry*>::iterator it = models.begin(); it != models.end(); it++){
		if (it->second->particleType() == BOUNDARY)
			radius = pspace * 0.255f;
		else
			radius = pspace * 0.51f;
		if (it->second->particleCollision(position, radius))
			return true;
	}
		

	return false;
}

void sphydrodynamics::initFluidParticle(size_t id, VEC3F& position)
{
	fluid_particle* p = &fp[id];
	p->setID(id);
	p->setType(FLUID);
	p->setPosition(position);
	p->setDensity(rho);
	p->setMass(particleMass[FLUID]);
	p->setPressure(0.f);
	p->setVelocity(VEC3F(0.0f, 0.0f, 0.0f));
}

size_t sphydrodynamics::makeFluid(VEC3F& source, bool onlyCountParticles)
{
// 	size_t cnt = 0;
// 	for (size_t x = 0; x < 40; x++){
// 		for (size_t y = 0; y < 80; y++){
// 			if (!onlyCountParticles)
// 				initFluidParticle(cnt, source + pspace * VEC3F((float)x, (float)y, 0.f));
// 			cnt++;
// 		}
// 	}
// 	return cnt;
std::stack<queuedParticle> analyzeQueue;
std::set<int> createdLocations;
queuedParticle q = { 0, 0, 0, 0, true, true };
analyzeQueue.push(q);
int hash;
if (particleCollision(source))
return 0;
while (!analyzeQueue.empty())
{
	q = analyzeQueue.top();
	analyzeQueue.pop();

	hash = utils::packIntegerPair(q.xMin, q.y);
	if (!createdLocations.count(hash))
	{
		if (!onlyCountParticles)
			initFluidParticle(createdLocations.size(), source + pspace * VEC3F((float)q.xMin, (float)q.y, 0.0f));
		createdLocations.insert(hash);

		if (!particleCollision(source + pspace * VEC3F((float)(q.xMin + 1), (float)q.y, 0.0f))){
			queuedParticle w = { q.xMin + 1, 0, q.y, 0, true, true };
			analyzeQueue.push(w);
		}
		if (!particleCollision(source + pspace *  VEC3F((float)(q.xMin - 1), (float)q.y, 0.0f)))
		{
			queuedParticle w = { q.xMin - 1, 0, q.y, 0, true, true };
			analyzeQueue.push(w);
		}
		if (!particleCollision(source + pspace *  VEC3F((float)q.xMin, (float)(q.y + 1), 0.0f)))
		{
			queuedParticle w = { q.xMin, 0, q.y + 1, 0, true, true };
			analyzeQueue.push(w);
		}
		if (!particleCollision(source + pspace *  VEC3F((float)q.xMin, (float)(q.y - 1), 0.0f)))
		{
			queuedParticle w = { q.xMin, 0, q.y - 1, 0, true, true };
			analyzeQueue.push(w);
		}
	}
}

return createdLocations.size();
}

size_t sphydrodynamics::initDummyCorner(size_t wallId, const VEC3F& pos, const VEC3F& n1, const VEC3F& n2, bool onlyCountParticles, bool isInner, bool isMov)
{
	size_t count = 0;
	int layers = (int)(fd->gridCellSize() / pspace);

	for (int i = 1; i <= layers; i++){

		float hp = rho * -9.80665f * i * (-pspace);
		VEC3F p1 = pos - (i * pspace) * n1;
		VEC3F p2 = pos - (i * pspace) * n2;

		count += 1;
		if (!onlyCountParticles){
			float dist1 = (p1 - pos).length();
			float dist2 = (p2 - pos).length();
			VEC3F norm1 = (p1 - pos) / dist1;
			VEC3F norm2 = (p2 - pos) / dist2;
			VEC3F p0 = (dist1 * norm1) + (dist2 * norm2) + pos;
			initDummyParticle(wallId + count, p0, hp, isInner, isInner, isMov);
		}

		if (isInner){
			continue;
		}
		count += 2;
		if (!onlyCountParticles)
		{
			hp = rho * -9.80665f * (p1.y - pos.y);
			initDummyParticle(wallId + count - 1, p1, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
			hp = rho * -9.80665f * (p2.y - pos.y);
			initDummyParticle(wallId + count, p2, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
		}

		if (i > 1){
			for (int j = 1; j < i; j++){
				count += 2;
				VEC3F p3 = p1 - (j * pspace) * n2;
				VEC3F p4 = p2 - (j * pspace) * n1;
				if (!onlyCountParticles){
					hp = rho * -9.80665f * (p3.y - pos.y);
					initDummyParticle(wallId + count - 1, p3, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
					hp = rho * -9.80665f * (p4.y - pos.y);
					initDummyParticle(wallId + count, p4, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
				}
			}
		}
	}

	return count;
}

// size_t sphydrodynamics::initDummyCorner3(size_t wallId, const VEC3F& pos, const VEC3F& n1, const VEC3F& n2, const VEC3F& n3, bool onlyCountParticles, bool isInner, bool isMov)
// {
// 	size_t count = 0;
// 	int layers = (int)(fd->gridCellSize() / pspace);
// 
// 	for (int i = 1; i <= layers; i++){
// 
// 		float hp = rho * -9.80665f * i * (-pspace);
// 		VEC3F p1 = pos - (i * pspace) * n1;
// 		VEC3F p2 = pos - (i * pspace) * n2;
// 
// 		count += 1;
// 		if (!onlyCountParticles){
// 			float dist1 = (p1 - pos).length();
// 			float dist2 = (p2 - pos).length();
// 			VEC3F norm1 = (p1 - pos) / dist1;
// 			VEC3F norm2 = (p2 - pos) / dist2;
// 			VEC3F p0 = (dist1 * norm1) + (dist2 * norm2) + pos;
// 			initDummyParticle(wallId + count, p0, hp, isInner, isInner, isMov);
// 		}
// 
// 		if (isInner){
// 			continue;
// 		}
// 		count += 2;
// 		if (!onlyCountParticles)
// 		{
// 			hp = rho * -9.80665f * (p1.y - pos.y);
// 			initDummyParticle(wallId + count - 1, p1, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
// 			hp = rho * -9.80665f * (p2.y - pos.y);
// 			initDummyParticle(wallId + count, p2, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
// 		}
// 
// 		if (i > 1){
// 			for (int j = 1; j < i; j++){
// 				count += 2;
// 				VEC3F p3 = p1 - (j * pspace) * n2;
// 				VEC3F p4 = p2 - (j * pspace) * n1;
// 				if (!onlyCountParticles){
// 					hp = rho * -9.80665f * (p3.y - pos.y);
// 					initDummyParticle(wallId + count - 1, p3, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
// 					hp = rho * -9.80665f * (p4.y - pos.y);
// 					initDummyParticle(wallId + count, p4, abs(hp) < 1e-9f ? 0.0f : hp, false, false, isMov);
// 				}
// 			}
// 		}
// 	}
// 
// 	return count;
// }


void sphydrodynamics::insertOverlappingLines(overlappingLine ol)
{
	for (int i = 0; i < overlappingLines.size(); i++){
		if ((ol.p1 - overlappingLines[i].p1).length() < 1e-9){
			if ((ol.p2 - overlappingLines[i].p2).length() < 1e-9){
				return;
			}
		}
		if ((ol.p2 - overlappingLines[i].p1).length() < 1e-9){
			if ((ol.p1 - overlappingLines[i].p2).length() < 1e-9){
				return;
			}
		}
	}
	overlappingLines.push_back(ol);
}

size_t sphydrodynamics::initOverlappingCorners(bool onlyCountParticles)
{
 	size_t count = 0;
	// 	size_t icount = 0;
	// 	size_t ccount = 0;
	if (tdim == DIM3){
		for (size_t i = 0; i < overlappingLines.size(); i++){
			overlappingLine *ol = &overlappingLines[i];
			VEC3F diff = ol->p2 - ol->p1;
			int lineCnt_a = (int)(diff.length() / particleSpacing() + 0.5f);
			float spacing_a = diff.length() / lineCnt_a;
			VEC3F unitdiff = diff.normalize();
			for (int i = 1; i < lineCnt_a; i++){
				VEC3F displacement = 0.f;
				if (!onlyCountParticles){
					displacement = ol->p1 + (i * spacing_a) * unitdiff;
					fluid_particle* p = &fp[overlappedCornersStart + count];
					p->setID(overlappedCornersStart + count);
					p->setIsCorner(true);
					p->setType(BOUNDARY);
					p->setPosition(displacement);
					p->setDensity(rho);
					p->setMass(particleMass[FLUID]);
					p->setPressure(0.f);
					p->setVelocity(VEC3F(0.f));
					particleCountByType[BOUNDARY]++;
				}
			//	count+=1;

				count += 1 + initDummyCorner(overlappedCornersStart + count, displacement, -ol->t1, -ol->t2, onlyCountParticles, false, false);
			}
		}
		int layers = (int)(fd->gridCellSize() / pspace);
		for (size_t i = 0; i < overlappingCorners.size(); i++){
			overlappingCorner *oc = &overlappingCorners[i];
			for (int j = 0; j <= layers; j++){
				VEC3F displacement = 0.f;
				if (!onlyCountParticles){
					displacement = oc->c1.position - j * pspace * oc->c3.normal;
					fluid_particle* p = &fp[overlappedCornersStart + count];
					p->setID(overlappedCornersStart + count);
					p->setIsCorner(true);
					p->setType(BOUNDARY);
					p->setPosition(displacement);
					p->setDensity(rho);
					p->setMass(particleMass[FLUID]);
					p->setPressure(0.f);
					p->setVelocity(VEC3F(0.f));
					particleCountByType[BOUNDARY]++;
				}
				count += 1 + initDummyCorner(overlappedCornersStart + count, displacement, oc->c1.normal, oc->c2.normal, onlyCountParticles, false, false);
				if(!oc->cnt){
					break;
				}
			}
		}
	}
	else{
		for (size_t i = 0; i < overlappingCorners.size(); i++){
			overlappingCorner *oc = &overlappingCorners[i];
			VEC3F tv = oc->c1.tangent - oc->c2.tangent;
			VEC3F tu = tv / tv.length();
			if (oc->isMovement){
				oc->sid = overlappedCornersStart + count;
			}
			if (!onlyCountParticles){
				fluid_particle* p = &fp[overlappedCornersStart + count];
				p->setID(overlappedCornersStart + count);
				if (oc->isInner){
					p->setIsFloating(oc->isInner);
				}
				p->setIsCorner(true);
				p->setType(BOUNDARY);
				p->setPosition(oc->c1.position);
				p->setDensity(rho);
				p->setTangent(tu);
				p->setNormal(oc->c1.normal);
				p->setNormal2(oc->c2.normal);
				p->setMass(particleMass[FLUID]);
				p->setPressure(0.f);
				p->setVelocity(VEC3F(0.0f, 0.0f, 0.0f));
				particleCountByType[BOUNDARY]++;
			}
			// 
			// 		if(oc.isInner && !onlyCountParticles)
			// 		{
			// 			InnerCornerDummyPressureIndex[icount * 5 + 0] = overlappedCornersStart + count;
			// 			icount++;
			// 		}

			float dot = oc->c1.normal.dot(oc->c2.tangent);
			//count+=1;
			if (dot <= 0 && tBT == DUMMY_PARTICLE_METHOD)
				count += 1 + initDummyCorner(overlappedCornersStart + count, oc->c1.position.toVector2(), oc->c1.normal.toVector2(), oc->c2.normal.toVector2(), onlyCountParticles, oc->isInner, oc->isMovement);
			else
				count += 1;
			if (oc->isMovement && onlyCountParticles){
				oc->cnt = (overlappedCornersStart + count) - oc->sid;
			}
		}
	}
	
	return count;
}

void sphydrodynamics::ComputeDeltaP()
{
	float QSq = pspace * pspace * skernel.h_inv_sq;
	if (tdim == DIM3)
		QSq *= pspace;
	deltap = sphkernel->sphKernel(QSq);
}

bool sphydrodynamics::preProcessGeometry()
{
	std::multimap<std::string, geo::geometry*>::iterator it;

	// find overlapping corners
	std::vector<corner> corners;
	std::vector<corner> corners2;
	bool iSc3 = false;
	for (it = models.begin(); it != models.end(); it++)
	{
		if (it->second->particleType() != BOUNDARY) // it's free surface geometry
			continue;
		std::vector<corner> objCorners = it->second->corners();
		for (size_t i = 0; i < objCorners.size(); i++)
		{
			for (size_t j = 0; j < corners.size(); j++)
			{

				if ((objCorners[i].position - corners[j].position).length() < 1e-9)
				{
					if (it->second->geometryType() == PLANE){
						for (size_t k = 0; k < overlappingCorners.size(); k++){
							overlappingCorner c = overlappingCorners[k];
							if ((c.c1.position - objCorners[i].position).length() < 1e-9){
								if ((c.c2.position - objCorners[i].position).length() < 1e-9){
									overlappingCorners[k].c3 = objCorners[i];
									overlappingCorners[k].cnt = 1;
									iSc3 = true;
								}
							}
						}
					}
					if (!iSc3){
						overlappingCorner c = { false, it->second->movement(), 0, 0, it->second->expressionVelocity(), objCorners[i], corners[j] };
						overlappingCorners.push_back(c);
					}
					else{
						iSc3 = false;
					}
					break;
// 					if (it->second->geometryType() == PLANE)
// 					{
// 						for (size_t k = 0; k < corners2.size(); k++){
// 							if ((corners[j].position - corners2[k].position).length() < 1e-9)
// 							{
// 								overlappingCorner c = { false, it->second->movement(), 0, 0, it->second->expressionVelocity(), objCorners[i], corners[j], corners2[k] };
// 							}
// 							corners2.push_back(corners[k]);
// 						}
// 					}
// 					else{
// 						bool isInner = false;
// 						if (it->second->geometryType() == SQUARE){
// 							isInner = true;
// 						}
// 						overlappingCorner c = { isInner, it->second->movement(), 0, 0, it->second->expressionVelocity(), objCorners[i], corners[j] };
// 						overlappingCorners.push_back(c);
// 						break;
// 					}					
				}
			}
			corners.push_back(objCorners[i]);
		}
	}

	for (int i = 0; i < (int)PARTICLE_TYPE_COUNT; i++)
		particleCountByType[i] = 0;

	np = makeFluid(ffloc, true);
	particleCountByType[FLUID] += np;
	size_t nfloat = makeFloating(true);
	particleCountByType[FLOATING] += nfloat;
	np += nfloat;
	if (tCorr == GRADIENT_CORRECTION){
		matKgc = new float6[np];
		memset(matKgc, 0, sizeof(float6) * np);
	}
	for (it = models.begin(); it != models.end(); it++){
		if (it->second->particleType() == BOUNDARY){
			it->second->sid = np;
			np += it->second->nParticle();
		}
	}

	if (numInnerCornerParticles){
		InnerCornerDummyPressureIndex = new size_t[5 * numInnerCornerParticles];
	}

	overlappedCornersStart = np;
	size_t overlapCount = initOverlappingCorners(true);
	if (numInnerCornerParticles){
		for (size_t i = 0; i < numInnerCornerParticles; i++)
		{
			InnerCornerDummyPressureIndex[i] += np;
		}
	}
	np += overlapCount;

	if (!np || !particleCountByType[FLUID])
	{

	}

	return true;
}

size_t sphydrodynamics::makeFloating(bool isOnlyCount)
{
	/*if (!isOnlyCount)*/
	size_t cnt = 0;
	size_t nfluid = particleCountByType[FLUID];
// 	for (size_t i = 0; i < 23; i++)
// 	{
// 		for (size_t j = i; j < 23-i; j++)
// 		{
// 			//bool fit = true;
// 			if (!isOnlyCount){
// 				size_t id = nfluid + cnt;
// 				VEC3F position = VEC3F(j * pspace + 0.225f, -1.0f * i * pspace + 0.3f, 0.f);
// 				fluid_particle *p = particle(id);
// 				p->setID(id);
// 				p->setType(FLOATING);
// 				p->setPosition(position);
// 				p->setDensity(rho);
// 				p->setMass(particleMass[FLUID]);
// 				p->setPressure(0.f);
// 				p->setVisible(false);
// 				p->setVelocity(VEC3F(0.0f, 0.0f, 0.0f));
// 				if (i == 0){
// 					p->setVisible(true);
// 				}
// 				if (j == i){
// 					p->setVisible(true);
// 					//fit = false;
// 				}
// 				if (j == 23 - i - 1)
// 					p->setVisible(true);
// 
// 			}
// 			cnt++;
// 		}
// 	}
// 	for (size_t i = 0; i < 20; i++){
// 		for (size_t j = 0; j < 20; j++){
// 			if (!isOnlyCount){
// 				size_t id = nfluid + cnt;
// 				VEC3F position = VEC3F(i * pspace + 0.35f, j * pspace + 0.03f, 0.f);
// 				fluid_particle *p = particle(id);
// 				p->setID(id);
// 				p->setType(FLOATING);
// 				p->setPosition(position);
// 				p->setDensity(rho*0.7f);
// 				p->setMass(particleMass[FLUID] * 0.7f);
// 				p->setPressure(0.f);
// 				p->setVisible(false);
// 				p->setVelocity(VEC3F(0.0f, 0.0f, 0.0f));
// 				if (i == 0 || i == 19)
// 					p->setVisible(true);
// 				if (j == 0 || j == 19)
// 					p->setVisible(true);
// 			}
// 			
// 			cnt++;
// 		}
// 	}
 	return cnt;
	//for (size_t i = 0; )
}

bool sphydrodynamics::initGeometry()
{
	makeFluid(ffloc, false);
	makeFloating(false);
	std::multimap<std::string, geo::geometry*>::iterator it;

	for (it = models.begin(); it != models.end(); it++)
		if (it->second->particleType() == BOUNDARY)
			it->second->build(false);

	initOverlappingCorners(false);

	//	sorter->sort();

	for (it = models.begin(); it != models.end(); it++){
		if (it->second->movement())
		{
// 			if (it->second->Type() == FLUID)
// 				it->second->InitExpression();//EnqueueSubprogram("init_" + it->second->Name());
// 			else
			it->second->initExpression();
		}
	}
	for (size_t i = 0; i < overlappingCorners.size(); i++){
		overlappingCorner oc = overlappingCorners[i];
		oc.isMovement = true;
// 		if (oc.isMovement)
// 		{
// // 			for (unsigned int j = 0; j < oc.cnt; j++){
// // 				fp[oc.sid + j].setVelocity(oc.iniVel);
// // 				fp[oc.sid + j].setAuxVelocity(oc.iniVel);
// // 			}
// 		}
	}
	// 	sorter->sort();

	return true;
}

void sphydrodynamics::gradientCorrection(size_t id, VEC3F& gradW)
{
	VEC3F _gW;
	if (tdim == DIM3){
		_gW.x = matKgc[id].s0 * gradW.x + matKgc[id].s1 * gradW.y + matKgc[id].s2 * gradW.z;
		_gW.y = matKgc[id].s1 * gradW.x + matKgc[id].s3 * gradW.y + matKgc[id].s4 * gradW.z;
		_gW.z = matKgc[id].s2 * gradW.x + matKgc[id].s4 * gradW.y + matKgc[id].s5 * gradW.z;
	}
	else{
		_gW.x = matKgc[id].s0 * gradW.x + matKgc[id].s1 * gradW.y;
		_gW.y = matKgc[id].s1 * gradW.x + matKgc[id].s2 * gradW.y;
		_gW.z = 0.f;
	}
	gradW = _gW;
}

// void sphydrodynamics::correctionGradient(size_t id, fluid_particle* parj, VEC3F& gradW, VEC3F& rba)
// {
// 	if (fp[id].particleType() != FLUID)
// 		return;
// 	float volj = parj->mass() / parj->density();
// 	VEC3F fr = volj * gradW;
// 	matKgc[id].s0 += fr.x * rba.x; matKgc[id].s1 += fr.x * rba.y; matKgc[id].s2 += fr.x * rba.z;
// 	matKgc[id].s1 += fr.y * rba.x; matKgc[id].s3 += fr.y * rba.y; matKgc[id].s4 += fr.y * rba.z;
// 	matKgc[id].s2 += fr.z * rba.x; matKgc[id].s4 += fr.z * rba.y; matKgc[id].s5 += fr.z * rba.z;
//}

float sphydrodynamics::newTimeStep()
{
	float dt_cfl = 0.1f * pspace / maxVel;
	float dt_forces = 0.25f*sqrt(skernel.h / maxAcc);
//	float dt_visc = 0.1f * pspace * pspace / dynVisc;
	return dt_cfl < dt_forces ? dt_cfl : dt_forces;// (dt_forces < dt_visc ? dt_forces : dt_visc);
}

void sphydrodynamics::calcFreeSurface(bool isInit)
{
	for (size_t i = 0; i < np; i++){
		fluid_particle *parI = particle(i);
		if (parI->particleType() == DUMMY){
			continue;
		}
		
		VEC3F posI = parI->position();
		float div_r = 0;

		for (NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			VEC3F posDif = posI - it->j->position();
			float gp = it->gradW.dot(posDif);
			div_r -= (it->j->mass() / it->j->density()) * gp;
		}

		parI->setDivP(div_r);
		if (div_r < fsFactor){
			parI->setFreeSurface(true);
			parI->setPressure(0.0f);
			//pressure[i] = 0.f;
		}
// 		else if (div_r < 1.8f && parI->particleType() == BOUNDARY){
// 			parI->setFreeSurface(true);
// 			parI->setPressure(0.0f);
// 		}
		else{
			parI->setFreeSurface(false);
		}
	}
	if (boundaryTreatment() == DUMMY_PARTICLE_METHOD){
		for (size_t i = 0; i < np; i++)
		{
			if (fp[i].particleType() == BOUNDARY){
				float press = fp[i].pressure();
				size_t j = i + 1;
				while (j < np && fp[j].particleType() == DUMMY){
					fp[j].setPressure(press);
					j++;
				}
			}
		}
	}
}

void sphydrodynamics::gradientCorrection()
{
	fluid_particle* _fp = NULL;
	VEC3F dp;
	float xx, yy, zz, xy, yx, xz, yz;
	float _xx, _yy, _xy;
	float6 imat = { 1.f, 0.f, 0.f, 1.f, 0.f, 1.f };
	float6 imat2d = { 1.f, 0.f, 1.f, 0.f, 0.f, 0.f };
	for (size_t i = 0; i < np; i++)
	{
		_fp = fp + i;
		if (_fp->particleType() == DUMMY)
			continue;
		xx = 0.f; yy = 0.f; zz = 0.f; xy = 0.f; xz = 0.f; yz = 0.f; yx = 0.f;
		_xx = 0.f; _yy = 0.f; _xy = 0.f;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			dp = _fp->position() - it->j->position();
			xx -= dp.x * it->gradW.x; 
			yy -= dp.y * it->gradW.y;
			//xy -= dp.y * it->gradW.x;
			//yx -= dp.x * it->gradW.y;
			//xy -= dp.y * it->gradW.x;
			xy -= dp.x * it->gradW.y;
			if (tdim == DIM3){
				zz -= dp.z * it->gradW.z;
				xz -= dp.x * it->gradW.z;
				yz -= dp.y * it->gradW.z;
			}
		}
		if (tdim == DIM3){
			float det = (_fp->mass() / rho) * (xx * (zz * yy - yz * yz)
										     - xy * (zz * xy - yz * xz)
											 + xz * (yz * xy - yy * xz));
			if (abs(det) > 0.01f){
				float6 corr = { (yy*zz - yz*yz) / det
					, (xz*yz - xy*zz) / det
					, (xy*yz - yy*xz) / det
					, (xx*zz - xz*xz) / det
					, (xy*xz - xx*yz) / det
					, (xx*yy - xy*xy) / det };
				matKgc[i] = corr;
			}
			else{
				matKgc[i] = imat;
			}
		}
		else{
			//xy = 0.5f * xy;
			float det = volume * (xx * yy - xy * xy);
			if (abs(det) > 0.01f/* && abs(xx) > 0.25f && abs(yy) > 0.25f*/){
				float6 corr = { yy / det
					, -xy / det
					//, -yx / det
					, xx / det
					, 0
					, 0
					,0};
				matKgc[i] = corr;
			}
			else{
				matKgc[i] = imat;
			}
		}
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
// 			if (_fp->particleType() == DUMMY)
// 				continue;
			gradientCorrection(i, it->gradW);
		}
// 		for (size_t i = 0; i < np; i++){
// 			fp[i].setPositionTemp(fp[i].position());
// 			fp[i].setVelocityTemp(fp[i].velocity());
// 			if (fp[i].particleType() == BOUNDARY){
// 				float press = fp[i].pressure();
// 				size_t j = i + 1;
// 
// 				while (j < np && fp[j].particleType() == DUMMY){
// 					fp[j].setPressure(press);
// 					j++;
// 				}
// 			}
// 		}
	}
}

bool sphydrodynamics::exportData(size_t part)
{
	char v;
	char pt[256] = { 0, };
	sprintf_s(pt, 256, "%s/part%04d.bin", (path + name).c_str() , part);
	std::fstream pf;
	pf.open(pt, std::ios::out | std::ios::binary);
	float ps;
	bool isf;
	for (size_t i = 0; i < np; i++){
		fluid_particle* _fp = fp + i;
		if (!_fp->IsVisible())
			continue;
		switch (_fp->particleType()){
		case FLUID: v = 'f'; pf.write(&v, sizeof(char)); break;
		case FLOATING: v = 'f'; pf.write(&v, sizeof(char)); break;
		case BOUNDARY: v = 'b'; pf.write(&v, sizeof(char)); break;
		case DUMMY: v = 'd'; pf.write(&v, sizeof(char)); break;
		}
		ps = _fp->pressure();
		isf = _fp->IsFreeSurface();
		pf.write((char*)&_fp->position(), sizeof(float) * 3);
		pf.write((char*)&_fp->velocity(), sizeof(float) * 3);
		pf.write((char*)&ps, sizeof(float));
		pf.write((char*)&isf, sizeof(bool));
// 		bool isv = _fp->IsVisible();
// 		pf.write((char*)&isv, sizeof(bool));
	}
	pf.close();
	return true;
}

bool sphydrodynamics::insertGhostParticle(size_t hash, fluid_particle& ng)
{
	std::map<size_t, std::list<fluid_particle>>::iterator it = ghosts.find(hash);
	if (hash == 0)
	{
		hash = 0;
	}
	if (it == ghosts.end())
	{
		//ng.setID(nRealParticle() + nghost);
		std::list<fluid_particle> ls;
		ls.push_back(ng);
		ghosts[hash] = ls;
		nghost++;
	}
	else
	{
		for (std::list<fluid_particle>::iterator n = it->second.begin(); n != it->second.end(); n++){
			if (n->position().x == ng.position().x && n->position().y == ng.position().y && n->position().z == ng.position().z)
				return false;
		}
		//ng.setID(nRealParticle() + nghost);
		it->second.push_back(ng);
		nghost++;
	}
	return true;
}

void sphydrodynamics::transferGhostParticle()
{
	size_t pnp = particleCountByType[FLUID] + particleCountByType[BOUNDARY];
	size_t cnt = 0;
	std::map<size_t, std::list<fluid_particle>>::iterator it = ghosts.begin();
	for (; it != ghosts.end(); it++){
		for (std::list<fluid_particle>::iterator n = it->second.begin(); n != it->second.end(); n++){
			n->setID(pnp + cnt);
			fp[pnp + cnt] = (*n);
			cnt++;
		}
	}
}

void sphydrodynamics::resizeParticle(size_t numg)
{
	size_t pnp = particleCountByType[FLUID] + particleCountByType[BOUNDARY];
	if (numg != 0)
	{
		fluid_particle *_fp = new fluid_particle[pnp + numg];
		for (size_t i = 0; i < pnp; i++){
			_fp[i] = fp[i];
		}
		//memcpy(_fp, fp, sizeof(fluid_particle) * pnp);
		delete[] fp;
		fp = new fluid_particle[pnp + numg];
		for (size_t i = 0; i < pnp; i++){
			fp[i] = _fp[i];
		}
	//	memcpy(fp, _fp, sizeof(fluid_particle) * pnp);
		np = pnp + numg;
	}
}

void sphydrodynamics::exportParticlePosition()
{
	std::fstream of;
	of.open("C:/C++/poswithghost.txt", std::ios::out);
	for (size_t i = 0; i < np; i++){
		of << fp[i].particleType() << " " << fp[i].IsFreeSurface() << " " << fp[i].position().x << " " << fp[i].position().y << " " << fp[i].position().z  << " " << fp[i].pressure() << std::endl;
	}
	of.close();
}

void sphydrodynamics::initializeGhostMap()
{
	for (std::map<size_t, std::list<fluid_particle>>::iterator it = ghosts.begin(); it != ghosts.end(); it++){
		it->second.clear();
	}
	ghosts.clear();
	nghost = 0;
}

void sphydrodynamics::runModelExpression(float dt, float time)
{
	std::multimap<std::string, geo::geometry*>::iterator it;
	for (it = models.begin(); it != models.end(); it++){
		if (it->second->movement()){
			it->second->runExpression(dt, time);
		}
	}

// 	for (size_t i = 0; i < overlappingCorners.size(); i++){
// 		overlappingCorner oc = overlappingCorners[i];
// 		if (oc.isMovement)
// 		{
// 			if (time >= 0.1f/* && time < 0.16f*/){
// 				float sign = 1.f;
// 				// 		if (time > 0.16f && time < 0.26f)
// 				// 			sign = -1.f;
// 				// 		else if (time > 0.26f)
// 				// 			sign = 1.f;
// 				if (time >= 0.1f && time < 0.2f)
// 					sign = -1.f;
// 				else if (time >= 0.2f && time < 0.3f)
// 					sign = 1.f;
// 				fluid_particle* fp = particle(oc.sid);
// 				for (unsigned int j = 0; j < oc.cnt; j++){
// 					fp = particle(oc.sid + j);
// 					VEC3F upos = fp->positionTemp();
// 					upos.x += 0.02f * sin(2.0f * M_PI * (time - 0.1f));//fp->position() + sign * dt * initVel;//abs(0.01f * sin(2.0f * (float)M_PI * (time - startMovementTime))) * VEC3F(1.0f, 0.0f, 0.0f); //dt * initVel;
// 					//fp->setPosition();
// 					//VEC3F uvel = abs(0.08f * M_PI * cos(4.0f * M_PI * (time - startMovementTime))) * VEC3F(1.f, 0.f, 0.f);//;*/ sign * initVel;
// 					VEC3F uvel = (upos - fp->position()) / dt;
// 					fp->setPosition(upos);
// 					//if (fp->particleType() == DUMMY){
// 					//fp->setVelocity(VEC3F(0.f, 0.f, 0.f));
// 					//fp->setAuxVelocity(VEC3F(0.f, 0.f, 0.f));
// 					//}
// 					//else{
// 					fp->setVelocity(uvel);
// 					fp->setAuxVelocity(uvel);
// 					//}
// 
// 					//fp->setPosition(upos);
// 				}
// 			}
// 		}
// 	}
}

float sphydrodynamics::updateTimeStep()
{
	float new_dt = 0.125f * skernel.h / maxVel;
// 	if (dt > new_dt)
// 		dt = new_dt;
	return new_dt;
}