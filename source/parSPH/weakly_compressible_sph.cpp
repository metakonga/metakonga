#include "weakly_compressible_sph.h"

using namespace parsph;

weakly_compressible_sph::weakly_compressible_sph(std::string _name)
	: sphydrodynamics(_name)
	, cs0(0)
	, tc_epsilon1(-0.2)
	, tc_epsilon2(0.01)
	, tVisco(ARTIFICIAL)
	, alphaViscosity(0.01)
	, betaViscosity(0)
	, rhop0(1000.0)
	, rhop0_inv(1.0/1000.0)
	, xsph_factor(0.5)
	, gamma(7)
	, CFLFactor(0.3)
	, mViscDt(0)
	, verletStep(0)
	, verletSteps(40)
{

}

weakly_compressible_sph::~weakly_compressible_sph()
{

}

bool weakly_compressible_sph::initialize()
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
	particleMass[BOUNDARY] = particleMass[FLUID]/* / 2*/;

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

	dist_epsilon = 0.001 * skernel.h * skernel.h;
	timestep_inv = 1.0 / timestep;
	kinematicViscosity = dynamicViscosity / density;

	freeSurfaceFactor = dimension == DIM2 ? 1.5 : 2.4;

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
	corr = dimension == DIM3 ? new double[particleCount * 8] : new double[particleCount * 4];
	free_surface = new bool[particleCount];
	memset(free_surface, 0, sizeof(bool) * particleCount);

	initGeometry();

	double maxHeight = 0;
	for(unsigned int i = 0; i < particleCount; i++){
		if(ps[i].Type() == BOUNDARY)
			particleCountByType[BOUNDARY]++;
		else if(ps[i].Type() == DUMMY)
			particleCountByType[DUMMY]++;
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
	
	B = rhop0 * cs0 * cs0 / gamma;
	//cs0 = sqrt((gamma * B) / rhop0);

	for(unsigned int i = 0; i < particleCount; i++){
		if(ps[i].Type() == BOUNDARY){
			double press = ps[i].Pressure();
			uint j = i + 1;
			while(j < particleCount && ps[j].Type() == DUMMY){
				ps[j].setPressure(press/* + ps[j].HydroPressure()*/);
				j++;
			}
		}
	}

	for(unsigned int i = 0; i < particleCount; i++){
		double _density = pow(((ps[i].Pressure() / B)+1), 1/gamma) * rhop0;
		ps[i].setDensity(_density);
		ps[i].setDensityOld(_density);
		ps[i].setMass(dimension == DIM2 ? particle_spacing * particle_spacing * _density : particle_spacing * particle_spacing * particle_spacing * _density);
	}

	sorter->sort();
	calcFreeSurface();
	return true;
}

void weakly_compressible_sph::copyFromCurrentToOld()
{
	for(unsigned int i = 0; i < particleCount; i++){
		ps[i].setPositionOld(ps[i].Position());
		ps[i].setVelocityOld(ps[i].Velocity());
		ps[i].setDensityOld(ps[i].Density());
	}
}

double weakly_compressible_sph::artificialCS(double cs)
{
	return cs0 * cs * cs * cs;
}

double weakly_compressible_sph::acceleration()
{
	s_particle* parI = NULL;
	double max_f = 0;
	double sigmaMax = 0;
	double viscdt = 0;
	double maxViscdt = 0;
	double csMax = 0;
	vector3<double> posI, velI, accI, posDif, velDif;
	double podsI, podsJ, densityI, tensileI, csI, csJ, densityAdd;
	for(unsigned int i = 0; i < particleCount; i++){
		parI = getParticle(i);
		if(parI->Type() != FLUID)
			continue;
		posI = parI->Position();
		velI = parI->Velocity();
		accI = vector3<double>(0.0, 0.0, 0.0);
		vector3<double> velXsph;
		densityI = parI->Density();
		podsI = parI->Pressure() / (densityI * densityI);
		
		tensileI = podsI * (podsI < 0 ? tc_epsilon1 : tc_epsilon2);
		csI = artificialCS(densityI * rhop0_inv);
		parI->setSoundOfSpeed(csI);
		if(csI > csMax)
			csMax = csI;
	
//double div_r = 0;
		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			posDif = posI - it->j->Position();
			velDif = velI - it->j->Velocity();
			//div_r -= (it->j->Mass() / it->j->Density()) * it->gradW.dot(posDif);
			podsJ = it->j->Pressure() / (it->j->Density() * it->j->Density());
			// tensile correction
			double f = it->W * deltaPKernelInv;
			f *= f; f *= f; // (Wij/Wdp)^4
			double tensileJ = podsJ * (podsJ < 0 ? tc_epsilon1 : tc_epsilon2);
			double sigma = abs(posDif.dot(velDif)) / (posDif.dot() + dist_epsilon);
			if(sigma > sigmaMax)
				sigmaMax = sigma;
			// Additional condition psitivie pressure tensile control only if both pressures positive
			double tensile = podsI < 0 ? tensileI : 0;
			tensile += podsJ < 0 ? tensileJ : 0;
			tensile += (podsI > 0 && podsJ > 0)  ? tensileI + tensileJ : 0;

			// pressure and viscosity acceleration
			switch(tVisco){
			case ARTIFICIAL:
				{
					densityAdd = 1 / (densityI + it->j->Density());
					csJ = artificialCS(it->j->Density() * rhop0_inv);
					viscdt = posDif.dot(velDif) / (posDif.dot() + dist_epsilon);
					double phi = skernel.h * viscdt;
					double visco = -8*((dynamicViscosity + parI->EddyViscosity()) + (dynamicViscosity + it->j->EddyViscosity())) * densityAdd * velDif.dot(posDif) / (posDif.dot() + dist_epsilon);
					accI -= it->j->Mass() * (podsI + podsJ + f * tensile + visco) * it->gradW;
					//accI -= (podsI + podsJ + f * tensile + (-alphaViscosity*phi*0.5*(csI + csJ))*densityAdd) * it->j->Mass() * it->gradW;
				}
				break;
			case LAMINAR:
				break;
			case SPS:
				break;
			}
			// XSPH
			velXsph += (it->j->Mass() * it->W * (densityAdd)) * velDif;
			maxViscdt = max(viscdt, maxViscdt);
		}
		parI->setAcceleration(accI + gravity);
		parI->setAuxiliaryVelocity(velI - 2 * xsph_factor * velXsph);
		double fa = accI.length();
		if(fa > max_f)
			max_f = fa;
// 
// 		parI->setDivP(div_r);
// 		if(div_r < freeSurfaceFactor){
// 			parI->setFreeSurface(true);
// 		}
// 		else{
// 			parI->setFreeSurface(false);
// 		}
	}
	return maxViscdt;
}

void weakly_compressible_sph::continuityEquation()
{
	s_particle* parI = NULL;
	double tmp = 0;
	for(unsigned int i = 0; i < particleCount; i++){
		parI = getParticle(i);
		vector3<double> velI = parI->Velocity();
		tmp = 0;
		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			vector3<double> velDif = velI - it->j->Velocity();
			tmp += it->j->Mass() * velDif.dot(it->gradW);
		}
		parI->setDensityDeriv(tmp);
	}
}

void weakly_compressible_sph::integratePredictor()
{
	s_particle* parI;
	for(unsigned int i = 0; i < particleCount; i++){
		parI = getParticle(i);
		parI->setDensity(parI->Density() + parI->DensityDeriv() * 0.5 * timestep);
		if(parI->Type() == FLUID){
			parI->setPosition(parI->Position() + 0.5 * timestep * parI->AuxiliaryVelocity());
			parI->setVelocity(parI->Velocity() + 0.5 * timestep * parI->Acceleration());
		}
	}
}

void weakly_compressible_sph::integrateCorrector()
{
	s_particle* parI;
	for(unsigned int i = 0; i < particleCount; i++){
		parI = getParticle(i);
		parI->setDensity(parI->DensityTemp() + parI->DensityDeriv() * timestep);
		if(parI->Type() == FLUID){
			parI->setPosition(parI->PositionTemp() + timestep * parI->AuxiliaryVelocity());
			parI->setVelocity(parI->VelocityTemp() + timestep * parI->Acceleration());
		}
	}

	for(unsigned int i = 0; i < particleCount; i++){
		parI = getParticle(i);
		parI->setDensity(2 * parI->Density() - parI->DensityTemp());
		if(parI->Type() == FLUID){
			parI->setPosition(2 * parI->Position() - parI->PositionTemp());
			parI->setVelocity(2 * parI->Velocity() - parI->VelocityTemp());
		}
	}
}

void weakly_compressible_sph::updatePosition()
{
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle *parI = getParticle(i);
		parI->PositionTemp() = parI->Position() + timestep * parI->VelocityTemp();
	}
}

void weakly_compressible_sph::eos()
{
	s_particle *parI;
	for(unsigned int i = 0; i < particleCount; i++){
		parI = getParticle(i);
		if(parI->Type() == DUMMY)
		{
			//parI->setPressure(0.0);
			continue;
		}
// 		if(parI->FreeSurface() && parI->Type() == FLUID)
// 			continue;
		double p = B * (pow(parI->Density() * rhop0_inv, 7) - 1.0);
		if(parI->FreeSurface() && parI->Type() == BOUNDARY)
			p = 0.0;
		parI->setPressure(p);

		if(parI->Type() == BOUNDARY){
			unsigned j = i+1;
			while(j < particleCount && ps[j].Type() == DUMMY){
				ps[j].setPressure(p/* + ps[j].HydroPressure()*/);
				j++;
			}
		}
	}
}

void weakly_compressible_sph::integrateVerlet()
{
	verletStep++;
	if(verletStep < verletSteps)
	{
		double twodt = timestep + timestep;
		double dtsq_05 = 0.5 * timestep * timestep;
		for(unsigned int i = 0; i < particleCount; i++){
			double newRhop = ps[i].DensityOld() + twodt * ps[i].DensityDeriv();
			ps[i].setDensityOld(ps[i].Density());
			ps[i].setDensity(newRhop);

			if(ps[i].Density() < rhop0 && ps[i].Type() != FLUID)
				ps[i].setDensity(rhop0);

			if(ps[i].Type() != FLUID)
				continue;

			ps[i].setPosition(ps[i].Position() + timestep * (ps[i].AuxiliaryVelocity()) + dtsq_05 * ps[i].Acceleration());
			vector3<double> newVel = ps[i].VelocityOld() + twodt * ps[i].Acceleration();
			ps[i].setVelocityOld(ps[i].Velocity());
			ps[i].setVelocity(newVel);
		}
	}
	else
	{
		double dtsq_05 = 0.5 * timestep * timestep;
		for(unsigned int i = 0; i < particleCount; i++){
			double newRhop = ps[i].Density() + timestep * ps[i].DensityDeriv();
			ps[i].setDensityOld(ps[i].Density());
			ps[i].setDensity(newRhop);

			if(ps[i].Density() < rhop0 && ps[i].Type() != FLUID)
				ps[i].setDensity(rhop0);

			if(ps[i].Type() != FLUID)
				continue;

			ps[i].setPosition(ps[i].Position() + timestep * (ps[i].AuxiliaryVelocity()) + dtsq_05 * ps[i].Acceleration());
			vector3<double> newVel = ps[i].Velocity() + timestep * ps[i].Acceleration();
			ps[i].setVelocityOld(ps[i].Velocity());
			ps[i].setVelocity(newVel);
		}
		verletStep = 0;
	}

}

void weakly_compressible_sph::CalculateDerivatives()
{
	acceleration();
	continuityEquation();
}

double weakly_compressible_sph::DtVariable()
{
	double fa_max = -FLT_MAX;
	double cs_max = -FLT_MAX;
	for(unsigned int i = 0; i < particleCountByType[FLUID]; i++){
		s_particle *parI = getParticle(i);
		double fa = parI->Acceleration().dot();
		if(fa_max <= fa)
			fa_max = fa;
		if(cs_max < parI->SoundOfSpeed())
			cs_max = parI->SoundOfSpeed();
	}
	double fa_sqrt = sqrt(sqrt(fa_max));
	double dt1 = sqrt(skernel.h) / fa_sqrt;
	double dt2 = skernel.h / (cs_max + skernel.h*mViscDt);
	double dt = CFLFactor * min(dt1, dt2);
	if(dt < 1e-7){
		dt = 1e-7;
	}
	return dt;
}

void weakly_compressible_sph::DensityByShepardFilter()
{
	for(unsigned int i = 0; i < particleCount; i++){
		s_particle* parI = getParticle(i);
		double densityI = 0.0;
		double weight = 0.0;
		if(parI->Density() < rhop0){
			bool pause = true;
		}
		for(NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			double mult = it->W * it->j->Mass();
			densityI += mult;
			weight += mult / it->j->Density();
		}
		weight += sphkernel->sphKernel(0) * parI->Mass() / parI->Density();
		densityI += sphkernel->sphKernel(0) * parI->Mass();
		densityI /= weight;

		parI->setDensity(densityI);
	}
}

double weakly_compressible_sph::run()
{
// 	for(unsigned int i = 0; i < particleCount; i++){
// 		if(i == 1912){
// 			bool pause = true;
// 		}
// 		s_particle* parI = getParticle(i);
// 		parI->setDensityTemp(parI->Density());
// 		parI->setVelocityTemp(parI->Velocity());
// 		parI->setPositionTemp(parI->Position());	
// 	}
// 	for(unsigned int i = 0; i < 2; i++){
// 		CalculateDerivatives();
// 		!i ? integratePredictor() : integrateCorrector();
// 		if(i == 0)
// 			sorter->sort();
// 		eos();
// 	}
	if(tturbulence)
		EddyViscosity();
 	mViscDt = acceleration();
 	continuityEquation();
 	//double dt_p = CFLCondition(mdt1);
	//timestep = DtVariable();
 	integrateVerlet();
	sorter->sort();
	calcFreeSurface();
	DensityByShepardFilter();
	eos();
 	//integratePredictor();
 	//sorter->sort();
 	
 	//// corrector step
 	//double mdt2 = acceleration();
 	//continuityEquation();
 	////double dt_c = CFLCondition(mdt2);
 	//integrateCorrector();
 	//sorter->sort();
 	//eos();
 	//double new_dt = min(mdt1, mdt2);
 	//timestep = mdt1 < timestep_min ? timestep_min : mdt1;
 	
 	//if(new_dt )
 	return timestep;
}

double weakly_compressible_sph::gpu_run()
{
	return 0;
}