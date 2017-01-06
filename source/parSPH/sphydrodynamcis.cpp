#include "sphydrodynamics.h"
#include "cu_sph_decl.cuh"
#include <cfloat>
#include <stack>
#include <queue>
#include <set>

using namespace parsph;

sphydrodynamics::sphydrodynamics(std::string _name)
	: ps(NULL)
	, name(_name)
	, corr(NULL)
	, matKgc(NULL)
	, free_surface(NULL)
	, sorter(NULL)
	, dimension(DIMENSION)
	, tturbulence(TURBULENCE)
	, device(DEVICE)
	, timestep_min(1e-5)
	// initialize device members
	, d_matKgc(NULL)
	, d_gamma(NULL)
	, d_sumKernel(NULL)
	, d_class(NULL)
	, d_pos(NULL)
	, d_vel(NULL)
	, d_auxVel(NULL)
	, d_gradP(NULL)
	, d_pressure(NULL)
	, d_density(NULL)
	, d_free_surface(NULL)
	, d_pos_temp(NULL)
	, d_auxPos(NULL)
	, d_iniPressure(NULL)
	, d_pressure_temp(NULL)
	, d_inner_particle(NULL)
	, d_hydro_pressure(NULL)
	, d_isFloating(NULL)
	, d_eddyViscosity(NULL)
	, InnerCornerDummyPressureIndex(NULL)
{
	numInnerCornerParticles = 0;
}

sphydrodynamics::~sphydrodynamics()
{
	if(corr) delete [] corr; corr = NULL;
	if(ps) delete [] ps; ps = NULL;
	if(sorter) delete sorter; sorter = NULL;
	if(matKgc) delete matKgc; matKgc = NULL;
	if(InnerCornerDummyPressureIndex) delete [] InnerCornerDummyPressureIndex; InnerCornerDummyPressureIndex = NULL;
	// dealloc device memory
	if(d_matKgc) cudaFree(d_matKgc); d_matKgc = NULL;
	if(d_isFloating) cudaFree(d_isFloating); d_isFloating = NULL;
	if(d_gamma) cudaFree(d_gamma); d_gamma = NULL;
	if(d_sumKernel) cudaFree(d_sumKernel); d_sumKernel = NULL;
	if(d_class) cudaFree(d_class); d_class = NULL;
	if(d_pos) cudaFree(d_pos); d_pos = NULL;
	if(d_vel) cudaFree(d_vel); d_vel = NULL;
	if(d_auxVel) cudaFree(d_auxVel); d_auxVel = NULL;
	if(d_gradP) cudaFree(d_gradP); d_gradP = NULL;
	if(d_pressure) cudaFree(d_pressure); d_pressure = NULL;
	if(d_density) cudaFree(d_density); d_density = NULL;
	if(d_free_surface) cudaFree(d_free_surface); d_free_surface = NULL;
	if(d_auxPos) cudaFree(d_auxPos); d_auxPos = NULL;
	if(d_pos_temp) cudaFree(d_pos_temp); d_pos_temp = NULL;
	if(d_vel_temp) cudaFree(d_vel_temp); d_vel_temp = NULL;
	if(d_iniPressure) cudaFree(d_iniPressure); d_iniPressure = NULL;
	if(d_pressure_temp) cudaFree(d_pressure_temp); d_pressure_temp = NULL;
	if(d_hydro_pressure) cudaFree(d_hydro_pressure); d_hydro_pressure = NULL;
	if(d_inner_particle) cudaFree(d_inner_particle); d_inner_particle = NULL;
	if(d_eddyViscosity) cudaFree(d_eddyViscosity); d_eddyViscosity = NULL;
}

void sphydrodynamics::setKernel(t_kernel _ker, bool correction, double h)
{
	skernel.kernel = _ker;
	skernel.correction = correction;
	skernel.h = h;
}

void sphydrodynamics::setBoundary(vector3<double> boundMin, vector3<double> boundMax)
{
	gridMin = boundMin;
	gridMax = boundMax;
	gridSize = boundMax - boundMin;
}

bool sphydrodynamics::initialize()
{
	std::cout << "Initializing the simulation" << std::endl;

	double supportRadius;
	switch(skernel.kernel){
	case QUADRATIC:
	case CUBIC_SPLINE:
	case GAUSS:
	case WENDLAND:
		supportRadius = 2 * skernel.h;
		break;
	case QUINTIC:
	case MODIFIED_GAUSS:
		supportRadius = 3 * skernel.h;
	}

	gridCellSize = supportRadius;

	if(preProcessGeometry())
		return false;

	double particleVolume = pow(particle_spacing, (int)dimension);
	particleMass[FLUID] = particleMass[DUMMY] = particleVolume * density;
	particleMass[BOUNDARY] = particleMass[FLUID] / 2;

	volume = particleVolume;
	kernelSupportRadius = supportRadius; 

	skernel.h_sq = skernel.h * skernel.h;
	skernel.h_inv = 1.0 / skernel.h;
	skernel.h_inv_sq = 1.0 / skernel.h / skernel.h;
	skernel.h_inv_2 = 1.0 / skernel.h / skernel.h;
	skernel.h_inv_3 = 1.0 / pow(skernel.h, 3);
	skernel.h_inv_4 = 1.0 / pow(skernel.h, 4);
	skernel.h_inv_5 = 1.0 / pow(skernel.h, 5);

	density_inv = 1.0 / density;
	density_inv_sq = 1.0 / (density * density);

	kinematicViscosity = dynamicViscosity / density;

	ComputeDeltaP();

	double dp = 1.0 / deltap;

 	return true;
}

void sphydrodynamics::textFileOut(double* odata, std::string fname, unsigned int n)
{
	double *data = NULL;
	if(device == GPU){
		data = new double[n];
		cudaMemcpy(data, odata, sizeof(double) * n, cudaMemcpyDeviceToHost);
	}
	else{
		data = odata;
	}
	std::fstream pf;
	pf.open(fname, std::ios::out);
	for(unsigned int i = 0; i < n; i++){
		pf << i << " " << data[i] << std::endl;
	}
	pf.close();
	if(device == GPU){
		delete [] data;
	}
}

void sphydrodynamics::textFileOut(double3* odata, std::string fname, unsigned int n)
{
	double3 *data = NULL;
	if(device == GPU){
		data = new double3[n];
		cudaMemcpy(data, odata, sizeof(double3) * n, cudaMemcpyDeviceToHost);
	}
	else{
		data = odata;
	}
	std::fstream pf;
	pf.open(fname, std::ios::out);
	for(unsigned int i = 0; i < n; i++){
		pf << i << " ---- " << data[i].x << " " << data[i].y << " " << data[i].z << std::endl;
	}
	pf.close();
	if(device == GPU){
		delete [] data;
	}
}

bool sphydrodynamics::preProcessGeometry()
{
	std::multimap<std::string, Geometry*>::iterator it;

	// find overlapping corners
	std::vector<Geometry::Corner> corners;
	for(it = models.begin(); it != models.end(); it++)
	{
		if(it->second->Type() != BOUNDARY) // it's free surface geometry
			continue;
		std::vector<Geometry::Corner> objCorners = it->second->Corners();
		for(unsigned int i = 0; i < objCorners.size(); i++)
		{
			for(unsigned int j = 0; j < corners.size(); j++)
			{
				if((objCorners[i].position - corners[j].position).length() < 1e-9)
				{
					bool isInner = false;
					if(it->second->GeometryType() == SQUARE){
						isInner = true;
					}
					OverlappingCorner c = {isInner, objCorners[i], corners[j]};
					overlappingCorners.push_back(c);
					break;
				}
			}
			corners.push_back(objCorners[i]);
		}
	}

	// count the particles
	for (int i = 0; i < (int)PARTICLE_TYPE_COUNT; i++)
		particleCountByType[i] = 0;

	particleCount = MakeFluid(floodFillLocation, true);
	particleCountByType[FLUID] += particleCount;
	if(correction == GRADIENT_CORRECTION){
		matKgc = new double6[particleCount];
		memset(matKgc, 0, sizeof(double6) * particleCount);
		if(device == GPU){
			checkCudaErrors( cudaMalloc((void**)&d_matKgc, sizeof(double6) * particleCount) );
			checkCudaErrors( cudaMalloc((void**)&d_gamma, sizeof(double3) * particleCount) );
			checkCudaErrors( cudaMalloc((void**)&d_sumKernel, sizeof(double) * particleCount) );
			checkCudaErrors( cudaMemcpy(d_matKgc, matKgc, sizeof(double6) * particleCount, cudaMemcpyHostToDevice) );
		}
	}

	for(it = models.begin(); it != models.end(); it++){
		if(it->second->Type() == BOUNDARY){
			it->second->startId = particleCount;
			particleCount += it->second->ParticleCount();
		}
	}

	if(numInnerCornerParticles){
		InnerCornerDummyPressureIndex = new unsigned int[5*numInnerCornerParticles];
	}

	overlappedCornersStart = particleCount;
	unsigned int overlapCount = initOverlappingCorners(true);
	if(numInnerCornerParticles){
		for(unsigned int i = 0; i < numInnerCornerParticles; i++)
		{
			InnerCornerDummyPressureIndex[i] += particleCount;
		}
	}
	particleCount += overlapCount;

	if(!particleCount || !particleCountByType[FLUID])
	{

	}

 	return true;
}

struct QueuedParticle 
{
	int xMin, xMax, y, upDownDir;
	bool goLeft, goRight;
};

unsigned int sphydrodynamics::MakeFluid(vector3<double> source, bool onlyCountParticles)
{
	std::stack<QueuedParticle> analyzeQueue;
	std::set<int> createdLocations;
	QueuedParticle q = {0, 0, 0, 0, true, true};
	analyzeQueue.push(q);
	int hash;
	if(particleCollision(source))
		return 0;
	while(!analyzeQueue.empty())
	{
		q = analyzeQueue.top();
		analyzeQueue.pop();

		hash = utils::packIntegerPair(q.xMin, q.y);
		if(!createdLocations.count(hash))
		{
			if(!onlyCountParticles)
				initFluidParticle(createdLocations.size(), source + particle_spacing * vector3<double>(q.xMin, q.y, 0.0));
			createdLocations.insert(hash);

			if(!particleCollision(source + particle_spacing * vector3<double>(q.xMin+1, q.y, 0.0))){
				QueuedParticle w = {q.xMin + 1, 0, q.y, 0, true, true};
				analyzeQueue.push(w);
			}
			if(!particleCollision(source + particle_spacing *  vector3<double>(q.xMin-1, q.y, 0.0)))
			{
				QueuedParticle w = {q.xMin-1, 0, q.y, 0, true, true};
				analyzeQueue.push(w);
			}    
			if(!particleCollision(source + particle_spacing *  vector3<double>(q.xMin, q.y+1, 0.0)))
			{
				QueuedParticle w = {q.xMin, 0, q.y+1, 0, true, true};
				analyzeQueue.push(w);
			}    
			if(!particleCollision(source + particle_spacing *  vector3<double>(q.xMin, q.y-1, 0.0)))
			{
				QueuedParticle w = {q.xMin, 0, q.y-1, 0, true, true};
				analyzeQueue.push(w);
			}
		}
	}

 	return createdLocations.size();
}

void sphydrodynamics::initFluidParticle(unsigned int id, vector3<double>& position)
{
	s_particle* p = &ps[id];
	p->setID(id);
	p->setType(FLUID);
	p->setPosition(position);
	p->setDensity(density);
	p->setMass(particleMass[FLUID]);
	p->setPressure(0.);
	p->setVelocity(vector3<double>(0.0, 0.0, 0.0));
}

bool sphydrodynamics::particleCollision(const vector3<double>& position)
{
	if(position.x > gridMax.x || position.y > gridMax.y)
		return true;
	if(position.x < gridMin.x || position.y < gridMin.y)
		return true;

	double radius = particle_spacing * 0.51;

	for(std::multimap<std::string,Geometry*>::iterator it=models.begin(); it != models.end(); it++)
		if(it->second->particleCollision(position, radius))
			return true;

 	return false;
}

bool sphydrodynamics::isCornerOverlapping(const vector3<double>& position)
{
	for(unsigned int i = 0; i < overlappingCorners.size(); i++){
		if((overlappingCorners[i].c1.position - position).length() < 1e-9)
			return true;
	}
 	return false;
}

unsigned int sphydrodynamics::initDummies(unsigned int wallId, const vector3<double>& pos, const vector3<double>& normal, bool onlyCountParticles, bool considerHp, int minusCount, bool isf)
{
	unsigned int layers = (unsigned int)(gridCellSize / particle_spacing);
	layers = layers - unsigned int(minusCount);
	if(!onlyCountParticles){
		for (unsigned int i=1; i<=layers; i++){
			double hp = considerHp ? density * -9.80665 * i * (-particle_spacing) : 0.0;
			if(considerHp)
			{
				bool pause = true;
			}
			initDummyParticle(wallId + i, pos - (i*particle_spacing) * normal, hp, false, isf);
		}
	}
 	return layers;
}

void sphydrodynamics::initDummyParticle(unsigned int id, const vector3<double>& position, double hydrop, bool isInner, bool isf)
{
	s_particle* p = &ps[id];
	p->setID(id);
	p->setType(DUMMY);
	p->setPosition(position);
	p->setDensity(density);
	p->setIsFloating(isf);
	p->setMass(particleMass[FLUID]);
	p->setPressure(0.);
	p->setHydroPressure(hydrop);
	p->setVelocity(vector3<double>(0.0, 0.0, 0.0));
	p->setIsInner(isInner);
}

unsigned int sphydrodynamics::initOverlappingCorners(bool onlyCountParticles)
{
	unsigned int count = 0;
// 	unsigned int icount = 0;
// 	unsigned int ccount = 0;
	for(unsigned int i = 0; i < overlappingCorners.size(); i++){
		OverlappingCorner oc = overlappingCorners[i];

		if(!onlyCountParticles){
			s_particle* p = &ps[overlappedCornersStart + count];
			p->setID(overlappedCornersStart + count);
			if(oc.isInner){
				p->setIsFloating(oc.isInner);
			}
			p->setType(BOUNDARY);
			p->setPosition(oc.c1.position);
			p->setDensity(density);
			p->setMass(particleMass[FLUID]);
			p->setPressure(0.);
			p->setVelocity(vector3<double>(0.0, 0.0, 0.0));
		}
// 
// 		if(oc.isInner && !onlyCountParticles)
// 		{
// 			InnerCornerDummyPressureIndex[icount * 5 + 0] = overlappedCornersStart + count;
// 			icount++;
// 		}

		double dot = oc.c1.normal.dot(oc.c2.tangent);

		if(dot <= 0)
			count += 1 + initDummyCorner(overlappedCornersStart + count, oc.c1.position.toVector2(), oc.c1.normal.toVector2(), oc.c2.normal.toVector2(), onlyCountParticles, oc.isInner);
	}
 	return count;

}

unsigned int sphydrodynamics::initDummyCorner(unsigned int wallId, const vector2<double>& pos, const vector2<double>& n1, const vector2<double>& n2, bool onlyCountParticles, bool isInner)
{
	unsigned int count = 0;
	int layers = (int)(gridCellSize / particle_spacing);

	for (int i = 1; i <= layers; i++){

		double hp = density * -9.80665 * i * (-particle_spacing);
		vector2<double> p1 = pos - n1 * (i * particle_spacing);
		vector2<double> p2 = pos - n2 * (i * particle_spacing);

		count += 1;
		if(!onlyCountParticles){
			double dist1 = (p1 - pos).length();
			double dist2 = (p2 - pos).length();
			vector2<double> norm1 = (p1 - pos) / dist1;
			vector2<double> norm2 = (p2 - pos) / dist2;
			vector2<double> p0 = (dist1 * norm1) + (dist2 * norm2) + pos;
			initDummyParticle(wallId + count, p0, hp, isInner, isInner);
		}

		if(isInner){
			continue;
		}
		count += 2;
		if(!onlyCountParticles)
		{
			hp = density * -9.80665 * (p1.y - pos.y);
			initDummyParticle(wallId + count - 1, p1, abs(hp) < 1e-9 ? 0.0 : hp, false, false);
			hp = density * -9.80665 * (p2.y - pos.y);
			initDummyParticle(wallId + count, p2, abs(hp) < 1e-9 ? 0.0 : hp, false, false);
		}

		if(i > 1){
			for(int j = 1; j < i; j++){
				count += 2;
				vector2<double> p3 = p1 - n2 * (j * particle_spacing);
				vector2<double> p4 = p2 - n1 * (j * particle_spacing);
				if(!onlyCountParticles){
					hp = density * -9.80665 * (p3.y - pos.y);
					initDummyParticle(wallId + count - 1, p3, abs(hp) < 1e-9 ? 0.0 : hp, false, false);
					hp = density * -9.80665 * (p4.y - pos.y);
					initDummyParticle(wallId + count, p4, abs(hp) < 1e-9 ? 0.0 : hp, false, false);
				}
			}
		}
	}

	return count;
}

void sphydrodynamics::setParticleShifting(bool enable, unsigned int frequency, double factor)
{
	particle_shifting.enable = enable;
	particle_shifting.frequency = frequency;
	particle_shifting.factor = factor;
}


void sphydrodynamics::ComputeDeltaP()
{
	double QSq = particle_spacing * particle_spacing * skernel.h_inv_sq;
	if(dimension == DIM3)
		QSq *= particle_spacing;
	deltap = sphkernel->sphKernel(QSq);
}

bool sphydrodynamics::initGeometry()
{
	MakeFluid(floodFillLocation, false);

	std::multimap<std::string,Geometry*>::iterator it;

	for(it=models.begin(); it != models.end(); it++)
		if(it->second->Type() == BOUNDARY)
			it->second->Build(false);

	initOverlappingCorners(false);

//	sorter->sort();

	for(it=models.begin(); it != models.end(); it++){
		if(it->second->ExpressionInit())
		{
			if(it->second->Type() == FLUID)
				it->second->InitExpression();//EnqueueSubprogram("init_" + it->second->Name());
			else
				it->second->InitExpression();
		}
	}

// 	sorter->sort();

 	return true;
}

void sphydrodynamics::kernelNormalize()
{

}

void sphydrodynamics::cu_gradientCorrection()
{
	cu_calcGradientCorrection(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_matKgc, d_gamma, d_sumKernel,  particleCount, d_density);
}

void sphydrodynamics::ClearMatKgc()
{
	const double6 zero_mat = {0, 0, 0, 0, 0, 0};
	for(unsigned int i = 0; i < particleCount; i++){
		if(ps[i].Type() == FLUID)
			matKgc[i] = zero_mat;
	}
}

void sphydrodynamics::CorrectionGradient(unsigned int id, s_particle* parj, vector3<double>& gradW, vector3<double>& rba)
{
	if(ps[id].Type() != FLUID)
		return;
	double volj = parj->Mass() / parj->Density();
	vector3<double> fr = volj * gradW;
	matKgc[id].s0 += fr.x * rba.x; matKgc[id].s1 += fr.x * rba.y; matKgc[id].s2 += fr.x * rba.z;
	matKgc[id].s1 += fr.y * rba.x; matKgc[id].s3 += fr.y * rba.y; matKgc[id].s4 += fr.y * rba.z;
	matKgc[id].s2 += fr.z * rba.x; matKgc[id].s4 += fr.z * rba.y; matKgc[id].s5 += fr.z * rba.z;	
}

void sphydrodynamics::invCorrectionGradient(unsigned int id)
{
// 	if(ps[id].Type() != FLUID)
// 		return;
// 	const symatrix imat = {1, 0, 0, 1, 0, 1};
// 	if(dimension == DIM2){
// 		matKgc[id].yy = 1.0;
// 	}
// 	symatrix mat = matKgc[id];
// 
// 	mat.xy *= 0.5;
// 	mat.xz *= 0.5;
// 	mat.yz *= 0.5;
// 	double det = mat.xx*mat.yy*mat.zz + 2.0*mat.xy*mat.yz*mat.xz - mat.xz*mat.yy*mat.xz - mat.xx*mat.yz*mat.yz - mat.xy*mat.xy*mat.zz; 
// 	if(abs(det)>0.01 && abs(mat.xx)>MATRIXKGC_CONTROL && abs(mat.yy)>MATRIXKGC_CONTROL && abs(mat.zz)>MATRIXKGC_CONTROL){
// 		symatrix invmat;
// 		invmat.xx=(mat.yy*mat.zz-mat.yz*mat.yz)/det;
// 		invmat.xy=(mat.xz*mat.yz-mat.xy*mat.zz)/det;
// 		invmat.xz=(mat.xy*mat.yz-mat.yy*mat.xz)/det;
// 		invmat.yy=(mat.xx*mat.zz-mat.xz*mat.xz)/det;
// 		invmat.yz=(mat.xy*mat.xz-mat.xx*mat.yz)/det;
// 		invmat.zz=(mat.xx*mat.yy-mat.xy*mat.xy)/det;
// 		matKgc[id]=invmat;
// 	}
// 	else{
// 		matKgc[id] = imat;
// 	}
}

vector3<double> sphydrodynamics::correctGradW(vector3<double> gradW, unsigned int i)
{
	if(dimension == DIM3){
		scalar8 *s8 = ((scalar8 *)corr) + i;
		return vector3<double>(
			gradW.x*s8->q + gradW.y*s8->w + gradW.z*s8->e,
			gradW.x*s8->w + gradW.y*s8->r + gradW.z*s8->a,
			gradW.x*s8->e + gradW.y*s8->a + gradW.z*s8->s);
	}
	scalar4 *s4 = ((scalar4 *)corr) + i;
 	return vector3<double>(gradW.x*s4->q + gradW.y*s4->w, gradW.x*s4->w + gradW.y*s4->e, 0.0);
}

bool sphydrodynamics::exportData()
{
	for(unsigned int i = 0; i < particleCount; i++){

	}
	return true;
}

void sphydrodynamics::runModelExpression(double dt, double time)
{
	std::multimap<std::string,Geometry*>::iterator it;
	for(it=models.begin(); it != models.end(); it++){
		if(it->second->ExpressionMovement())
		{
			if(it->second->Type() == FLUID)
				it->second->InitExpression();//EnqueueSubprogram("init_" + it->second->Name());
			else
				it->second->RunExpression(dt, time);
		}
	}
}

double sphydrodynamics::CFLCondition(double mdt1)
{
	double mdt2 = min(mdt1, 0.125 * (skernel.h_sq / dynamicViscosity));
	return mdt2;
}

void sphydrodynamics::cu_updateBodyForceAndMoment(Geometry* Geo)
{
	//vector3<double> *h_pos = new vector3<double>[particleCount];
	//vector3<double> *h_gradP = new vector3<double>[particleCount];
	//checkCudaErrors( cudaMemcpy(h_pos, d_pos, sizeof(vector3<double>) * particleCount, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy(h_gradP, d_gradP, sizeof(vector3<double>) * particleCount, cudaMemcpyDeviceToHost) );
	if(Geo->GeometryType() == SQUARE){
		geo::Square *sqr = dynamic_cast<geo::Square*>(Geo);
		vector3<double> spoint[4] = { sqr->edge[3], sqr->edge[0], sqr->edge[1], sqr->edge[2] };
		vector3<double> epoint[4] = { sqr->edge[0], sqr->edge[1], sqr->edge[2], sqr->edge[3] };
		vector3<double> normal[4] = { sqr->normal[0], sqr->normal[1], sqr->normal[2], sqr->normal[3] };
		//double3 *d_sp, d_ep;
		double3 *d_Pf;
		//unsigned int *d_segIndex;
		double3* d_spoint;
		double3* d_epoint;
		double3* d_normal;
		unsigned int* seg_n;
		checkCudaErrors( cudaMalloc((void**)&seg_n, sizeof(unsigned int)*400)); 
		checkCudaErrors( cudaMalloc((void**)&d_Pf, sizeof(double3)*400));
		checkCudaErrors( cudaMemset(d_Pf, 0, sizeof(double3)*400) );
		checkCudaErrors( cudaMemset(seg_n, 0, sizeof(unsigned int)*400));
		//checkCudaErrors( cudaMalloc((void**)&d_sp, sizeof(double3) * 400) );
		//checkCudaErrors( cudaMalloc((void**)&d_ep, sizeof(double3) * 400) );
		checkCudaErrors( cudaMalloc((void**)&d_spoint, sizeof(double3)*4) );
		checkCudaErrors( cudaMalloc((void**)&d_epoint, sizeof(double3)*4) );
		checkCudaErrors( cudaMalloc((void**)&d_normal, sizeof(double3)*4) );

		//checkCudaErrors( cudaMalloc((void**)&d_segIndex, sizeof(unsigned int) * particleCount) );
		cudaMemcpy(d_spoint, spoint, sizeof(double3)*4, cudaMemcpyHostToDevice);
		cudaMemcpy(d_epoint, epoint, sizeof(double3)*4, cudaMemcpyHostToDevice);
		cudaMemcpy(d_normal, normal, sizeof(double3)*4, cudaMemcpyHostToDevice);
	//	for(int i = 0; i < 4; i++){
		//int i = 0;
			cu_findLineSegmentIndex(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_pos, d_gradP, d_pressure, d_isFloating, d_Pf, d_spoint, d_epoint, d_normal, seg_n, particleCount);
	//	}
		vector3<double> *h_Pf = new vector3<double>[400];
		unsigned int *h_segn = new unsigned int[400];

		checkCudaErrors( cudaMemcpy(h_Pf, d_Pf, sizeof(double3)*400, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(h_segn, seg_n, sizeof(unsigned int)*400, cudaMemcpyDeviceToHost) );
		//textFileOut(d_Pf, "C:/C++/ghfgh.txt", 4);
		vector3<double> bodyForce;     
		
		for(unsigned int i = 0; i < 4; i++){
			double dl = (epoint[i] - spoint[i]).length()/100;
			for(unsigned int j = 0; j < 100; j++){
				if(h_segn[i*100+j]){
					bodyForce += (dl/h_segn[i*100 +j]) * h_Pf[i*100 + j];
				}
				
			}
		}
// 		double dl = particle_spacing;
// 		for(unsigned int i = 0; i < 4; i++){
// 			if(h_segn[i]){
// 				bodyForce.x += dl * h_Pf[i].x / h_segn[i];
// 				bodyForce.y += dl * h_Pf[i].y / h_segn[i];
// 				bodyForce.z += dl * h_Pf[i].z / h_segn[i];
// 			}
// 		}

		std::cout << "bodyForce : [" << bodyForce.x << " " << bodyForce.y << " " << bodyForce.z << "]" <<std::endl;
		cu_updateBodyInformation(d_class, d_pos, d_vel, make_double3(bodyForce.x, bodyForce.y, bodyForce.z), d_isFloating, d_spoint, d_epoint, particleCount);
		cudaMemcpy(spoint, d_spoint, sizeof(double3)*4, cudaMemcpyDeviceToHost);
		cudaMemcpy(epoint, d_epoint, sizeof(double3)*4, cudaMemcpyDeviceToHost);
		sqr->edge[3] = spoint[0]; sqr->edge[0] = spoint[1]; sqr->edge[1] = spoint[2]; sqr->edge[2] = spoint[3];
		//vector3<double> epoint[4] = { sqr->edge[0], sqr->edge[1], sqr->edge[2], sqr->edge[3] };

		delete [] h_Pf;
		cudaFree(d_Pf);
		cudaFree(d_spoint);
		cudaFree(d_epoint);
		cudaFree(d_normal);
		cudaFree(seg_n);
	}
}

void sphydrodynamics::cu_eddyViscosity()
{
	cu_calcEddyViscosity(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_vel, d_density, d_eddyViscosity, particleCount);
}

void sphydrodynamics::EddyViscosity()
{
	vector3<double> posDif;
	vector3<double> velDif;
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle* parI = getParticle(i);
// 		if(parI->Type() == DUMMY){
// 			parI->setEddyViscosity(0.0);
// 			continue;
// 		}
		double s2 = 0.0;
		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			posDif = parI->Position() - it->j->Position();
			velDif = parI->Velocity() - it->j->Velocity();
			double rhopbar = (parI->Density() + it->j->Density()) / (parI->Density() * it->j->Density());
			s2 += rhopbar * (velDif.dot() / (posDif.dot() + dist_epsilon)) * posDif.dot(it->gradW);
		}
		s2 *= -0.5;
		parI->setEddyViscosity(particle_spacing * particle_spacing * sqrt(s2));
	}
		
}

void sphydrodynamics::calcFreeSurface()
{	
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle *parI = getParticle(i);
		if(parI->Type() == DUMMY){
			continue;
		}
		if(parI->IsFloating()){
			bool pause = true;
		}
		vector3<double> posI = parI->Position();
		double div_r = 0;

		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			vector3<double> posDif = posI - it->j->Position();
			div_r -= (it->j->Mass() / it->j->Density()) * it->gradW.dot(posDif);
		}

		parI->setDivP(div_r);
		if(div_r < freeSurfaceFactor){
			parI->setFreeSurface(true);
			parI->setPressure(0.0);
		}
		else{
			parI->setFreeSurface(false);
		}
	}
}