#include "incompressible_sph.h"
#include "cu_sph_decl.cuh"
#include <iomanip>

#include "cublas.h"

using namespace parsph;
 
incompressible_sph::incompressible_sph(std::string _name)
 	: sphydrodynamics(_name)
	, volumes(NULL)
	, rhs(NULL)
//	, temp_vector_memory(NULL)
	, strong_dirichlet(false)
	, projectionOrder(1)
	, d_rhs(NULL)
	, d_lhs(NULL)
	, d_residual(NULL)
	, d_conjugate0(NULL)
	, d_conjugate1(NULL)
	, d_tmp0(NULL)
	, d_tmp1(NULL)
{

}

incompressible_sph::~incompressible_sph()
{
	if(rhs) delete [] rhs; rhs = NULL;
	if(free_surface) delete [] free_surface; free_surface = NULL;
	if(volumes) delete [] volumes; volumes = NULL;
	// gpu
	if(d_rhs) checkCudaErrors( cudaFree(d_rhs) ); d_rhs = NULL;
	if(d_lhs) checkCudaErrors( cudaFree(d_lhs) ); d_lhs = NULL;
	if(d_residual) checkCudaErrors( cudaFree(d_residual) ); d_residual = NULL;
	if(d_conjugate0) checkCudaErrors( cudaFree(d_conjugate0) ); d_conjugate0 = NULL;
	if(d_conjugate1) checkCudaErrors( cudaFree(d_conjugate1) ); d_conjugate1 = NULL;
	if(d_tmp0) checkCudaErrors( cudaFree(d_tmp0) ); d_tmp0 = NULL;
	if(d_tmp1) checkCudaErrors( cudaFree(d_tmp1) ); d_tmp1 = NULL;
}

bool incompressible_sph::initialize()
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

	if(!preProcessGeometry())
		return false;

	double particleVolume = pow(particle_spacing, (int)dimension);
	particleMass[FLUID] = particleMass[DUMMY] = particleVolume * density;
	particleMass[BOUNDARY] = particleMass[FLUID];

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

	dist_epsilon = 0.01 * skernel.h * skernel.h;
	timestep_inv = 1.0 / timestep;
	kinematicViscosity = dynamicViscosity / density;

	freeSurfaceFactor = dimension == DIM2 ? 1.53 : 2.4;

	sorter = new grid(this);

	sorter->initGrid();

	switch(skernel.kernel){
	case QUINTIC:
		sphkernel = new quintic(this);
		break;
	case CUBIC_SPLINE:
		sphkernel = new cubic_spline(this);
		break;
	}

	ComputeDeltaP();

	deltaPKernelInv = 1.0 / deltap;

	//classes = new char[particleCount];
	ps = new s_particle[particleCount];
	volumes = new double[particleCount];
	corr = dimension == DIM3 ? new double[particleCount * 8] : new double[particleCount * 4];
	free_surface = new bool[particleCount];
	rhs = new double[particleCount];
	lhs = new double[particleCount];
	memset(free_surface, 0, sizeof(bool) * particleCount);

  	initGeometry();
	std::multimap<std::string,Geometry*>::iterator it;
	for(it=models.begin(); it != models.end(); it++)
		if(it->second->Type() == BOUNDARY)
			it->second->InitExpressionDummyParticles();

	double maxHeight = 0;
	for(unsigned int i = 0; i < particleCount; i++){
		if(ps[i].Type() != FLUID)
			continue;
		if(maxHeight < ps[i].Position().y)
			maxHeight = ps[i].Position().y;
		if(ps[i].Type() == DUMMY)
			ps[i].setPressure(0.0);
	}
	for(unsigned int i = 0; i < particleCount; i++){
 		if(ps[i].Type() == DUMMY){
 			continue;
		}
		double press0 = ps[i].Density() * gravity.length() * (maxHeight - ps[i].Position().y);
		ps[i].setPressure(press0);
		ps[i].setVelocityOld(ps[i].Velocity());
	}
	
	sorter->sort();
	calcFreeSurface();

	for(unsigned int i = 0; i < particleCount; i++){
		if(ps[i].Type() == BOUNDARY){
			double press = ps[i].Pressure();
			uint j = i + 1;
			while(j < particleCount && ps[j].Type() == DUMMY){
				ps[j].setPressure(press + ps[j].HydroPressure());
				j++;
			}
		}
	}

// 	for(unsigned int i = particleCountByType[FLUID]; i < particleCount; i++){
// 		s_particle* parI = getParticle(i);		
// 		if(parI->IsInner() && parI->Type() == DUMMY){
// 			double sumP = 0;
// 			for(NeighborInnerIterator it = parI->NeighborsInner().begin(); it != parI->NeighborsInner().end(); it++){
// 				s_particle *parJ = getParticle((*it));
// 				sumP += parJ->Pressure();
// 			}
// 			parI->setPressure(sumP / parI->NeighborsInner().size());
// 		}			
// 	}

	if(device == GPU){
		t_particle* h_class = new t_particle[particleCount];
		double3* h_pos = new double3[particleCount];
		double3* h_vel = new double3[particleCount];
		double* h_pressure = new double[particleCount];
		double* h_hpressure = new double[particleCount];
		double* h_density = new double[particleCount];
		bool* h_innerp = new bool[particleCount];
		bool* h_isFloating = new bool[particleCount];
		for(unsigned int i = 0; i < particleCount; i++){
			h_innerp[i] = false;
			s_particle* parI = getParticle(i);
			h_class[i] = parI->Type();
			h_pos[i] = make_double3(
				static_cast<double>(parI->Position().x), 
				static_cast<double>(parI->Position().y),
				static_cast<double>(parI->Position().z));
			h_vel[i] = make_double3(
				static_cast<double>(parI->Velocity().x), 
				static_cast<double>(parI->Velocity().y), 
				static_cast<double>(parI->Velocity().z));
			h_pressure[i] = static_cast<double>(parI->Pressure());
			h_hpressure[i] = parI->HydroPressure();
			h_innerp[i] = parI->IsInner();
			h_isFloating[i] = parI->IsFloating();
			if(h_isFloating[i]){
				h_vel[i] = make_double3(0.0, 0.0, 0.0);
			}
			//h_density[i] = density;
		}
		if(tturbulence){
			checkCudaErrors( cudaMalloc((void**)&d_eddyViscosity, sizeof(double)*particleCount) );
		}
		checkCudaErrors( cudaMalloc((void**)&d_inner_particle, sizeof(bool)* particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_free_surface, sizeof(bool) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_rhs, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_lhs, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMemset(d_lhs, 0, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_class, sizeof(t_particle) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_pos, sizeof(double3) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_auxPos, sizeof(double3) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_vel, sizeof(double3) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_auxVel, sizeof(double3) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_divP, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_gradP, sizeof(double3) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_pressure, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_hydro_pressure, sizeof(double) * particleCount));
		checkCudaErrors( cudaMalloc((void**)&d_residual, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_conjugate0, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_conjugate1, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_tmp0, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_tmp1, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_iniPressure, sizeof(double) * particleCount) );
		checkCudaErrors( cudaMalloc((void**)&d_isFloating, sizeof(bool) * particleCount) );

		checkCudaErrors( cudaMemcpy(d_class, h_class, sizeof(t_particle) * particleCount, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_pos, h_pos, sizeof(double3)*particleCount, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_vel, h_vel, sizeof(double3)*particleCount, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_pressure, h_pressure, sizeof(double)*particleCount, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_iniPressure, d_pressure, sizeof(double)*particleCount, cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(d_hydro_pressure, h_hpressure, sizeof(double)*particleCount, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_inner_particle, h_innerp, sizeof(bool)*particleCount, cudaMemcpyHostToDevice) );
		checkCudaErrors( cudaMemcpy(d_isFloating, h_isFloating, sizeof(bool)*particleCount, cudaMemcpyHostToDevice) );

		if(particle_shifting.enable){
			checkCudaErrors( cudaMalloc((void**)&d_pos_temp, sizeof(double3)* particleCount) );
			checkCudaErrors( cudaMalloc((void**)&d_vel_temp, sizeof(double3)* particleCount) );
			checkCudaErrors( cudaMalloc((void**)&d_pressure_temp, sizeof(double)* particleCount) );
		}

		device_parameters cte = {
			correction,
			dimension,
			skernel.kernel,
			particleCount,
			sorter->Cells(),
			make_double3((double)sorter->GridMin().x, (double)sorter->GridMin().y, (double)sorter->GridMin().z),
			make_double3((double)sorter->GridMax().x, (double)sorter->GridMax().y, (double)sorter->GridMax().z),
			make_double3((double)sorter->GridSize().x, (double)sorter->GridSize().y, (double)sorter->GridSize().z),
			make_double3((double)gravity.x, (double)gravity.y, (double)gravity.z),
			(double)density,
			(double)sphkernel->KernelConst(),
			(double)sphkernel->KernelGradConst(),
			(double)sphkernel->KernelSupport(),
			(double)sphkernel->KernelSupprotSq(),
			(double)skernel.h,
			(double)particle_spacing,
			make_double3((double)kernelSupportRadius.x, (double)kernelSupportRadius.y, (double)kernelSupportRadius.z),
			(double)deltaPKernelInv,
			(double)gridCellSize,
			make_int3(sorter->CellCounts().x, sorter->CellCounts().y, sorter->CellCounts().z),
			(double)(1.0 / gridCellSize),
			(double)particleMass[FLUID], 
			(double)0.06,
			(double)dynamicViscosity,
			(double)dist_epsilon,
			(double)skernel.h_sq,
			(double)skernel.h_inv,
			(double)skernel.h_inv_sq,
			(double)skernel.h_inv_2,
			(double)skernel.h_inv_3,
			(double)skernel.h_inv_4,
			(double)skernel.h_inv_5,
			(double)timestep,
			(double)density_inv,
			(double)density_inv_sq,
			(double)timestep_inv,
			(double)kinematicViscosity,
			(double)freeSurfaceFactor,
			(double)particle_shifting.factor
		};

		setSymbolicParameter(&cte);

		delete [] h_pos;
		delete [] h_vel;
		delete [] h_pressure;
		delete [] h_hpressure;
		delete [] h_innerp;
		delete [] h_isFloating;
	}
	//double pp = ps[5941].Pressure();
	return true;
}

void incompressible_sph::tempPositions()
{
	for(unsigned int i = 0; i < particleCountByType[FLUID]; i++){
		ps[i].Position() += timestep * ps[i].Velocity();
		ps[i].setPositionTemp(ps[i].Position());
	}
}

void incompressible_sph::tempVelocities()
{
	double minf=FLT_MAX;
	
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle *parI = getParticle(i);
		if(parI->Type() != FLUID){
			continue;
		}
		vector3<double> posI = parI->Position();
		vector3<double> velI = parI->Velocity();
		vector3<double> accI = vector3<double>(0, 0, 0);
		//double div_r = 0;

		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			vector3<double> posDif = posI - it->j->Position();
			vector3<double> velDif = velI - it->j->Velocity();
			//div_r -= (it->j->Mass() / it->j->Density()) * it->gradW.dot(posDif);
			if(skernel.correction){
				
			}
			accI += 8 * it->j->Mass() * ( (/*dynamicViscosity + */dynamicViscosity) / (parI->Density() + it->j->Density()) ) * ( velDif.dot(posDif) / (posDif.dot() + dist_epsilon) ) * it->gradW;
		}
		accI += gravity;
// 		if(accI.length() < minf){
// 			minf = accI.length();
// 		}
		parI->setAuxiliaryVelocity(velI + timestep * accI);
		//pf << parI->AuxiliaryVelocity().x << " " << parI->AuxiliaryVelocity().y << " " << parI->AuxiliaryVelocity().z << std::cout;
	//	parI->setDivP(div_r);
// 		if(div_r < freeSurfaceFactor){
// 			free_surface[i] = 1;
// 		}
// 		else{
// 			free_surface[i] = 0;
// 		}
	}
	//min_f = minf;
}

double incompressible_sph::dot(double *vec1, double *vec2)
{
	double sum = 0;
	for(unsigned int i = 0; i < particleCount; i++){
		if(ps[i].Type() != DUMMY/* || !ps[i].FreeSurface()*/)
			sum += vec1[i] * vec2[i];
	}
	return sum;
}

void incompressible_sph::correctorStep()
{
	double maxv = 0;
	//double minf = FLT_MAX;
 	for(unsigned int i = 0; i < particleCount; i++){
		s_particle *parI = getParticle(i);
		if(parI->Type() != FLUID)
			continue;

		double pI = parI->Pressure() / ( parI->Density() * parI->Density() );
		double tensileI = pI * (pI < 0 ? -0.2 : 0.01);
		vector3<double> gradP;

		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			double f = it->W * deltaPKernelInv;
			f*=f; f*=f;
			double pJ = it->j->Pressure() / ( it->j->Density() * it->j->Density() );
			double tensileJ = pJ * (pJ < 0 ? -0.2 : 0.01);
			double tensile = tensileI + tensileJ;
			gradP += it->j->Mass() * (pI + pJ/* + f*tensile*/) * it->gradW;
		}
		vector3<double> accI = gradP / parI->Density();

		if(i == 237)
		{
			bool pause = true;
		}
		parI->setVelocity(parI->AuxiliaryVelocity() - timestep * accI);
		parI->setPosition(parI->Position() + timestep * parI->Velocity());
// 		if(parI->Velocity().length() > maxv){
// 			maxv = parI->Velocity().length();
// 		}
 	}
//	max_vel = maxv;
}

bool incompressible_sph::solvePressureWithBiCGSTAB()
{
 	double *residual = new double[particleCount];
 	double *conjugate0 = new double[particleCount];
	double *conjugate1 = new double[particleCount];
	double *tmp0 = new double[particleCount];
	double *tmp1 = new double[particleCount];
 	double ip_rr = 0;
	for(unsigned int i = 0; i < particleCount; i++){
		residual[i] = 0.0;
		conjugate0[i] = 0.0;
		conjugate1[i] = 0.0;
		tmp0[i] = 0.0;
		tmp1[i] = 0.0;
	}
	pressurePoissonEquation(tmp0, NULL);
	textFileOut(rhs, "C:/C++/rhs.txt", particleCount);
	textFileOut(tmp0, "C:/C++/tmp0.txt", particleCount);
	for(unsigned int i = 0; i < particleCount; i++){
		residual[i] = rhs[i] = rhs[i] - tmp0[i];
		conjugate0[i] = residual[i]; 
		
		if(ps[i].Type() != DUMMY)
			ip_rr += residual[i] * residual[i];
	}
	if(abs(ip_rr) <= DBL_EPSILON)
	{
		return true;
	}
 	double norm_sph_squared = ip_rr;
 	double residual_norm_squared;

	double alpha = 0;
	double omega = 0;
	double beta = 0;

 	for(unsigned int i = 0; i < ppeMaxIteration; i++){
		//std::cout << "iteration : " << i;
		pressurePoissonEquation(tmp0, conjugate0);
		alpha = ip_rr / dot(rhs, tmp0);
		for(unsigned int j = 0; j < particleCount; j++){
			conjugate1[j] = residual[j] - alpha * tmp0[j];
		}
		pressurePoissonEquation(tmp1, conjugate1);
		omega = dot(tmp1, conjugate1) / dot(tmp1, tmp1);
		for(unsigned int j = 0; j < particleCount; j++){
			if(ps[j].FreeSurface() || ps[j].Type() == DUMMY){
				residual[j] = 0.0;
				ps[j].setPressure(0.0);
				continue;
			}
			double press = getParticle(j)->Pressure();
			press += alpha * conjugate0[j] + omega * conjugate1[j];
			residual[j] = conjugate1[j] - omega * tmp1[j];
			getParticle(j)->setPressure(press);
		}
		residual_norm_squared = dot(residual, residual);

		//std::cout << "   norm : " << residual_norm_squared / norm_sph_squared << std::endl;
		if(abs(residual_norm_squared / norm_sph_squared) <= solvingTolerance * solvingTolerance)
			break;

		double new_ip_rr = dot(residual, rhs);
		beta = (new_ip_rr / ip_rr) * (alpha / omega);
		ip_rr = new_ip_rr;
		for(unsigned int j = 0; j < particleCount; j++){
			conjugate0[j] = residual[j] + beta * (conjugate0[j] - omega * tmp0[j]);
		}
	}

	for(unsigned int i = 0; i < particleCount; i++){
		t_particle type = ps[i].Type();
		if(ps[i].FreeSurface()){
			ps[i].setPressure(0.0);
			continue;
		}
		if(type == BOUNDARY)
		{
			double press = ps[i].Pressure();
			double hpress = 0.0;
			unsigned int j = i + 1;
			while(j < particleCount && ps[j].Type() == DUMMY){
				hpress = ps[j].HydroPressure();
				ps[j++].setPressure(press/* + hpress*/);
			}
		}
	}

	delete [] residual;
	delete [] conjugate0;
	delete [] conjugate1;
	delete [] tmp0;
	delete [] tmp1;
  	return true;
}

void incompressible_sph::shiftParticles()
{
	vector3<double> posDif;
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle *parI = getParticle(i);
		t_particle type = parI->Type();
		if(type != FLUID || free_surface[i]){
			continue;
		}

		vector3<double> posI = parI->PositionTemp();
		double effRadiusSq = sphkernel->KernelSupprotSq() * skernel.h_sq;

		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			unsigned int j = it->j->ID();
			posDif = posI - it->j->PositionTemp();
			if(free_surface[j]){
				double distSq = posDif.dot();
				if(distSq < effRadiusSq)
					effRadiusSq = distSq;
			}
		}

		int neighborCount = 0;
		double avgSpacing = 0;
		vector3<double> shiftVec = vector3<double>(0,0,0);

		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			unsigned int j = it->j->ID();
			posDif = posI - it->j->PositionTemp();
			double distSq = posDif.dot();
			if(distSq > effRadiusSq)
				continue;
			double dist = sqrt(distSq);
			neighborCount++;
			avgSpacing += dist;
			shiftVec += posDif / (distSq * dist);
		}

		if(!neighborCount)
			continue;

		avgSpacing /= neighborCount;
		shiftVec *= avgSpacing * avgSpacing;

		double velMagnitude = parI->VelocityTemp().length();
		shiftVec = min(shiftVec.length() * particle_shifting.factor * velMagnitude * timestep, particle_spacing) * shiftVec.normalize();
		parI->Position() += shiftVec;
	}
}

void incompressible_sph::shiftParticlesUpdate()
{
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle *parI = getParticle(i);
		t_particle type = parI->Type();
		if(type != FLUID || free_surface[i])
			continue;

		vector3<double> posI = parI->PositionTemp();
		vector3<double> gp = vector3<double>(0, 0, 0);
		vector3<double> gvx = vector3<double>(0, 0, 0);
		vector3<double> gvy = vector3<double>(0, 0, 0);
		vector3<double> velI = parI->VelocityTemp();
		double pI = parI->PressureTemp();

		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			unsigned int j = it->j->ID();
			vector3<double> velJ = it->j->VelocityTemp();
			vector3<double> gradW = (it->j->Mass() / it->j->Density()) * it->gradW;
			gp += (it->j->PressureTemp() + pI) * gradW;
			gvx += (velJ.x - velI.x) * gradW;
			gvy += (velJ.y - velI.y) * gradW;
		}
		vector3<double> dr = parI->Position() - posI;
		parI->Pressure() += gp.dot(dr);
		parI->Velocity() += vector3<double>(gvx.dot(dr), gvy.dot(dr), 0.0);
	}
}

void incompressible_sph::predictionStep()
{
	vector3<double> mvdif[30] = {0.f, };
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle *parI = getParticle(i);
		if(parI->Type() != FLUID){
			rhs[i] = 0;
			continue;
		}
		vector3<double> posI = parI->Position();
		vector3<double> velI = parI->Velocity();

		double div_u = 0;
		unsigned int ii = 0;
		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			vector3<double> velDif = velI - it->j->Velocity();
			mvdif[ii++] = velDif;
			div_u += it->j->Mass() * velDif.dot(it->gradW);
		}

		rhs[i] = -div_u * timestep_inv;
	}
}

void incompressible_sph::pressurePoissonEquation(double *out, double *vec)
{
	for(unsigned int i = 0; i < particleCount; i++){
		double press=0;
		s_particle *parI = getParticle(i);
		if(parI->Type() == DUMMY){
			out[i] = 0.0;
			continue;
		}
		if(parI->FreeSurface()){
			out[i] = 0.0;
			continue;
		}

		vector3<double> posI = parI->Position();
		vector3<double> velI = parI->Velocity();
		double pressI = vec ? vec[i] : parI->Pressure();
		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			vector3<double> posDif = posI - it->j->Position();
			double pressDif = pressI - ( vec ? vec[it->j->ID()] : it->j->Pressure() );
			double _press = 0;
			_press = it->j->Mass() * pressDif * ( posDif.dot(it->gradW) ) / (posDif.dot() + dist_epsilon);
			press += _press;
		}
		press *= 2 / (parI->Density()*parI->Density());

		out[i] = press;
	}
}

double incompressible_sph::CFLcondition()
{
	double min_value = min(0.4 * skernel.h / max_vel, 0.25 * sqrt(skernel.h / (min_f + 1e-10)));
	return min(min_value, 0.125 * skernel.h_sq / dynamicViscosity);
}

void incompressible_sph::cu_tempPositions()
{
	cu_auxiliaryPosition(d_class, d_pos, d_vel, d_auxPos, d_isFloating, particleCount);
}

void incompressible_sph::cu_tempVelocities()
{
	cu_auxiliaryVelocity(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_vel, d_auxVel, d_isFloating, d_eddyViscosity, d_matKgc, d_gamma, d_sumKernel, particleCount);
}

void incompressible_sph::cu_freeSurface()
{
	cu_calcFreeSurface(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_free_surface, d_divP, particleCount);
}

void incompressible_sph::cu_predictionStep()
{
// 	textFileOut(d_pos, "C:/C++/d_pos.txt", particleCount);
// 	textFileOut(d_auxPos, "C:/C++/d_auxPos.txt", particleCount);
// 	textFileOut(d_auxVel, "C:/C++/d_auxVel.txt", particleCount);
	cu_predictor(d_class, d_free_surface, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_auxVel, d_rhs, d_isFloating, d_matKgc, d_gamma, d_sumKernel, particleCount);
}

void incompressible_sph::cu_pressurePoissonEquation()
{
	cu_PPEquation(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_pos, d_vel, d_pressure, d_lhs, d_matKgc, d_hydro_pressure, particleCount);
}

void incompressible_sph::cu_PPESolver()
{
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate_v2(&handle);
	double ip_rr = 0;
	//checkCudaErrors( cudaMemcpy(d_pressure, d_iniPressure, sizeof(double)*particleCount, cudaMemcpyDeviceToDevice) );
	cu_PPEquation_PPESolver(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_pressure, d_lhs, d_matKgc, d_gamma, d_sumKernel, d_free_surface, particleCount);
	//  	textFileOut(d_rhs, "C:/C++/drhs.txt", particleCount);
	//  	textFileOut(d_lhs, "C:/C++/dtmp0.txt", particleCount);
	ip_rr = initPPESolver(d_class, d_rhs, d_lhs, d_residual, d_conjugate0, d_conjugate1, d_tmp0, d_tmp1, d_free_surface, particleCount);
	cublasDcopy_v2(handle, particleCount, d_residual, 1, d_conjugate0, 1);

	if (abs(ip_rr) <= DBL_EPSILON)
	{
		return;
	}
	double norm_sph_squared = ip_rr;
	double residual_norm_squared;
	double alpha = 0;
	double omega = 0;
	double beta = 0;
	double malpha = 0;
	unsigned int i = 0;
	for(; i < ppeMaxIteration; i++){
		//std::cout << "iteration : " << i << std::endl;
		//std::cout << "step 1" << std::endl;
		cu_dummyScalarCopy(d_class, d_conjugate0, particleCount);
		//std::cout << "step 2" << std::endl;
		//double* h_c0 = new double;
		//cudaMemcpy(h_c0, d_conjugate0, sizeof(double), cudaMemcpyDeviceToHost);
		cu_PPEquation_PPESolver(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_conjugate0, d_tmp0, d_matKgc, d_gamma, d_sumKernel, d_free_surface, particleCount);
		//cudaMemcpy(h_c0, d_conjugate0, sizeof(double), cudaMemcpyDeviceToHost);
// 		std::cout << "step 3" << std::endl;
// 		double *h_tmp0 = new double[particleCount];
// 		cudaMemcpy(h_tmp0, d_tmp0, sizeof(double)*particleCount, cudaMemcpyDeviceToHost);
// 		std::fstream pf;
// 		pf.open("C:/C++/d_tmp0.txt", std::ios::out);
// 		for (size_t j = 0; j < particleCount; j++){
// 			pf << i << " " << h_tmp0[j] << std::endl;
// 		}
// 		pf.close();
		//double dot1 = cu_dot(d_rhs, d_tmp0);
		alpha = ip_rr / cu_dot(d_rhs, d_tmp0); malpha = -alpha;
		cublasDcopy_v2(handle, particleCount, d_residual, 1, d_conjugate1, 1);
		cublasDaxpy_v2(handle, particleCount, &malpha, d_tmp0, 1, d_conjugate1, 1);
		cu_dummyScalarCopy(d_class, d_conjugate1, particleCount);
		//cudaMemcpy(h_c0, d_conjugate1, sizeof(double), cudaMemcpyDeviceToHost);
		cu_PPEquation_PPESolver(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_auxPos, d_conjugate1, d_tmp1, d_matKgc, d_gamma, d_sumKernel, d_free_surface, particleCount);
		omega = cu_dot(d_tmp1, d_conjugate1) / cu_dot(d_tmp1, d_tmp1);
		cu_updatePressureAndResidual(alpha, d_conjugate0, omega, d_conjugate1, d_tmp1, d_pressure, d_residual, d_free_surface, d_class, particleCount);
		residual_norm_squared = cu_dot(d_residual, d_residual);

		if (abs(residual_norm_squared / norm_sph_squared) <= solvingTolerance * solvingTolerance)
			break;

		double new_ip_rr = cu_dot(d_residual, d_rhs);
		beta = (new_ip_rr / ip_rr) * (alpha / omega);
		ip_rr = new_ip_rr;
		cu_updateConjugate(d_conjugate0, d_residual, d_tmp0, beta, omega, d_free_surface, d_class, particleCount);
	}
	//std::cout << "total iteration : " << i << std::endl;
	//cu_dummyScalarCopy(d_class, d_pressure, particleCount);
	cu_setPressureFreesurfaceAndDummyParticle(d_class, d_free_surface, d_pressure, d_hydro_pressure, particleCount);
	//textFileOut(d_pressure, "C:/C++/d_pressure.txt", particleCount);
	cu_setInnerParticlePressureForDummyParticle(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_pos, d_pressure, d_inner_particle, particleCount);
	cublasDestroy_v2(handle);
}

double incompressible_sph::cu_dot(double* d1, double* d2)
{
	return dot6(d_class, free_surface, d1, d2, particleCount);
}

void incompressible_sph::cu_correctStep()
{
	cu_corrector(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_pos, d_auxPos, d_vel, d_auxVel, d_gradP, d_pressure, d_isFloating, d_matKgc, d_gamma, d_sumKernel, particleCount);
}

void incompressible_sph::cu_shiftParticles()
{
	cu_shifting(d_class, sorter->cuHashes(), sorter->cuCellStart(), d_free_surface, d_pos_temp, d_vel_temp, d_pressure_temp, d_pos, d_vel, d_pressure, d_matKgc, particleCount);
}

double incompressible_sph::run()
{
	sorter->sort();
	tempVelocities();
	calcFreeSurface();
	predictionStep();

	/*pressurePoissonEquation();*/
	solvePressureWithBiCGSTAB();
	correctorStep();

	if(particle_shifting.enable && (timestep_count + 1) % particle_shifting.frequency == 0){
		sorter->sort();
		for(unsigned int i = 0; i < particleCount; i++){
			ps[i].setPositionTemp(ps[i].Position());
			ps[i].setVelocityTemp(ps[i].Velocity());
			ps[i].setPressureTemp(ps[i].Pressure());
		}
		shiftParticles();
		shiftParticlesUpdate();
	}
	timestep_count++;
 	return timestep;
}

double incompressible_sph::gpu_run()
{
	cu_tempPositions();
	sorter->cusort();
	if(tturbulence)
		cu_eddyViscosity();
	//textFileOut(d_eddyViscosity, "C:/C++/eddyVisc.txt", particleCount);
	if(correction == GRADIENT_CORRECTION){
		cu_gradientCorrection();
	}
	cu_tempVelocities();
	cu_freeSurface();
	cu_predictionStep();
	//cu_pressurePoissonEquation();
	cu_PPESolver();
	cu_correctStep();

	std::multimap<std::string,Geometry*>::iterator it;
	for(it=models.begin(); it != models.end(); it++)
		if(it->second->Type() == BOUNDARY && it->second->GeometryType() == SQUARE)
			cu_updateBodyForceAndMoment(it->second);

	if(particle_shifting.enable && (timestep_count + 1) % particle_shifting.frequency == 0){
		sorter->cusort();
		checkCudaErrors( cudaMemcpy(d_pos_temp, d_pos, sizeof(double3)*particleCount, cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(d_vel_temp, d_vel, sizeof(double3)*particleCount, cudaMemcpyDeviceToDevice) );
		checkCudaErrors( cudaMemcpy(d_pressure_temp, d_pressure, sizeof(double)*particleCount, cudaMemcpyDeviceToDevice) );
		cu_shiftParticles();
	}
// 	double3 *h_pos = new double3[particleCount];
// 	checkCudaErrors( cudaMemcpy(h_pos, d_pos, sizeof(double3) * particleCount, cudaMemcpyDeviceToHost) );
// 	std::fstream pf;
// 	pf.open("C:/C++/POSITION.txt", std::ios::out);
// 	for(unsigned int i = 0; i < particleCount; i++){
// 		//std::cout << std::setprecision(15) << h_pos[7776].x << std::endl;
// 		pf << h_pos[i].x << " " << h_pos[i].y << " " << h_pos[i].z << std::endl;
// 	}
// 	pf.close();
// 	delete [] h_pos;

	timestep_count++;
	return timestep;
}