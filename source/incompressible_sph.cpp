#include "incompressible_sph.h"
#include "CSKernel.h"
#include "quinticKernel.h"
#include "quadraticKernel.h"
#include "timer.h"
#include <fstream>
#include <iomanip>

incompressible_sph::incompressible_sph(std::string _path, std::string _name)
	: sphydrodynamics(_path, _name)
{
	
}

incompressible_sph::~incompressible_sph()
{
// 	if (conjugate0) delete[] conjugate0; conjugate0 = NULL;
// 	if (conjugate1) delete[] conjugate1; conjugate1 = NULL;
// 	if (tmp0) delete[] tmp0; tmp0 = NULL;
// 	if (tmp1) delete[] tmp1; tmp1 = NULL;
// 	if (residual) delete[] residual; residual = NULL;
}

bool incompressible_sph::initialize()
{
	std::cout << "Initializing the simulation" << std::endl;

	float supportRadius;
	switch (skernel.kernel){
	case QUADRATIC:
		supportRadius = 2.5f * skernel.h;
		break;
	case CUBIC_SPLINE:
	case GAUSS:
	case WENDLAND:
		supportRadius = 2.f * skernel.h;
		break;
	case QUINTIC:
	case MODIFIED_GAUSS:
		supportRadius = 3.f * skernel.h;
	}

	fd->setGridCellSize(supportRadius);

	if (!preProcessGeometry())
		return false;

	float particleVolume = pow(pspace, (int)tdim);
	particleMass[FLUID] = particleMass[DUMMY] = particleVolume * rho;
	particleMass[BOUNDARY] = particleMass[FLUID];

	volume = particleVolume;
	ksradius = supportRadius;

	skernel.h_sq = skernel.h * skernel.h;
	skernel.h_inv = 1.0f / skernel.h;
	skernel.h_inv_sq = 1.0f / (skernel.h * skernel.h);
	skernel.h_inv_2 = 1.0f / skernel.h / skernel.h;
	skernel.h_inv_3 = 1.0f / pow(skernel.h, 3);
	skernel.h_inv_4 = 1.0f / pow(skernel.h, 4);
	skernel.h_inv_5 = 1.0f / pow(skernel.h, 5);

	rho_inv = 1.0f / rho;
	rho_inv_sq = 1.0f / (rho * rho);

	depsilon = 0.01f * skernel.h * skernel.h;
	dt_inv = 1.0f / dt;
	kinVisc = dynVisc / rho;

	fsFactor = tdim == DIM2 ? 1.5f : 2.4f;

	fd->initGrid();

	switch (skernel.kernel){
	case QUINTIC:
		sphkernel = new quinticKernel(this);
		break;
	case QUADRATIC:
		sphkernel = new quadraticKernel(this);
		break;
	case CUBIC_SPLINE:
		sphkernel = new CSKernel(this);
		break;
	}

	ComputeDeltaP();

	deltaPKernelInv = 1.0f / deltap;

	//classes = new char[particleCount];

	fp = new fluid_particle[np];
	volumes = new float[np];
	corr = tdim == DIM3 ? new float[np * 8] : new float[np * 4];
	//free_surface = new bool[np];
	rhs = new float[np];
// 	lhs = new float[np];
// 	//pressure = new float[np];
// 	conjugate0 = new float[np];
// 	conjugate1 = new float[np];
// 	tmp0 = new float[np];
// 	tmp1 = new float[np];
// 	residual = new float[np];

	initGeometry();
	std::multimap<std::string, geo::geometry*>::iterator it;
	//exportParticlePosition();
	float maxHeight = 0.f;
	for (size_t i = 0; i < np; i++){
		if (fp[i].particleType() != FLUID)
			continue;
		if (maxHeight < fp[i].position().y)
			maxHeight = fp[i].position().y;
		if (fp[i].particleType() == DUMMY)
			fp[i].setPressure(0.0);
	}
	for (size_t i = 0; i < np; i++){
		fp[i].setAuxPosition(fp[i].position());
		fp[i].setPositionOld(fp[i].position());
		if (fp[i].particleType() == DUMMY){
			continue;
		}
		float press0 = fp[i].density() * grav.length() * (maxHeight - fp[i].position().y);
		//	pressure[i] = press0;
 		fp[i].setHydroPressure(press0);
 		fp[i].setPressure(press0);
		
		fp[i].setVelocityOld(fp[i].velocity());
	}

	fd->sort(true);
	calcFreeSurface(true);
	exportParticlePosition();
	if (boundaryTreatment() == DUMMY_PARTICLE_METHOD){
		for (size_t i = 0; i < np; i++){
			fp[i].setPositionOld(fp[i].position());
			fp[i].setVelocityOld(fp[i].velocity());
			if (fp[i].particleType() == BOUNDARY){
				float press = fp[i].pressure();
				size_t j = i + 1;
				if (press == 0.f)
					press = 0.f;
				while (j < np && fp[j].particleType() == DUMMY){
					fp[j].setPressure(press);
					j++;
				}
			}
		}
	}
	else{
		for (size_t i = 0; i < np; i++){
			fp[i].setPositionTemp(fp[i].position());
			fp[i].setVelocityTemp(fp[i].velocity());
		}
	}
	

	if (fs.is_open()){
		fs << "num_fluid " << particleCountByType[FLUID] + particleCountByType[FLOATING] << std::endl
			<< "num_boundary " << particleCountByType[BOUNDARY] << std::endl
			<< "num_dummy " << particleCountByType[DUMMY] << std::endl;
	}
	

	return true;
}

void incompressible_sph::auxiliaryPosition()
{
	fluid_particle *_fp = NULL;
	for (size_t i = 0; i < nRealParticle(); i++){
		_fp = fp + i;

		if (_fp->particleType() == FLUID){
			_fp->setPosition(_fp->positionTemp());
			_fp->setVelocity(_fp->velocityTemp());
			_fp->setAuxPosition(_fp->position() + dt * _fp->velocity());// = _fp->position() + dt * _fp->velocity();
		}		
		else{
			if (_fp->velocity().x != 0.f)
				bool pause = true;
		}
	}
}

void incompressible_sph::auxiliaryVelocity()
{

	fluid_particle *_fp = NULL;
	VEC3F ip, iv, ia, dp, dv;
	//VEC3I cell, loopStart, loopEnd;
	float p_1 = 0, p_2 = 0;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		ip = _fp->position();
		iv = _fp->velocity();
		ia = grav;
// 		if (_fp->particleType() != FLUID){
// 			_fp->setAuxVelocity(VEC3F(0.f));
// 			continue;
// 		}
		if (_fp->particleType() == DUMMY && boundaryTreatment() == GHOST_PARTICLE_METHOD)
		{
			_fp->setAuxVelocity(-fp[_fp->baseFluid()].auxVelocity());
			continue;
		}
		if (_fp->particleType() != FLUID)
		{
			//_fp->setVelocity(VEC3F(0.f));
			//_fp->setAuxVelocity(VEC3F(0.f));
			continue;
		}
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (it->j->particleType() == BOUNDARY)
				continue;
			dp = ip - it->j->position();
			dv = iv - it->j->velocity();
// 			p_1 = (rho * dynVisc + rho * dynVisc) / (rho * rho);
// 			p_2 = dp.dot(it->gradW) / (dp.dot(dp) + depsilon);
// 			ia += it->j->mass() * (p_1 * p_2) * dv;
			p_1 = 8*(dynVisc + dynVisc) / (rho + rho);
			p_2 = dv.dot(dp) / (dp.dot(dp) + depsilon);
			ia += it->j->mass() * (p_1 * p_2) * it->gradW;
		}
		_fp->setAuxVelocity(iv + dt * ia);
	}
}

void incompressible_sph::predictionStep()
{
	fluid_particle *_fp = NULL;
	VEC3F ip, iv, dp, dv;
	VEC3I cell, loopStart, loopEnd;
	float div_u = 0.f;
	if (rhs)
		delete[] rhs;
	rhs = new float[np];
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		if (_fp->particleType() == DUMMY){
			rhs[i] = 0.f;
			continue;
		}
		
		div_u = 0.0f;
		ip = _fp->auxPosition();
		iv = _fp->auxVelocity();
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if(it->j->particleType() == BOUNDARY)
				continue;
			dp = ip - it->j->auxPosition();
			dv = iv - it->j->auxVelocity();
			div_u += it->j->mass() * dv.dot(it->gradW);
		}
		rhs[i] = -div_u * dt_inv;
	}
} 

void incompressible_sph::ghostDummyScalarSet(float* src)
{
	fluid_particle *_fp = NULL;
	for (size_t j = 0; j < np; j++){
		_fp = fp + j;
// 		if (_fp->IsFreeSurface() && _fp->particleType() != DUMMY){
// 			_fp->setPressure(0.f);
// 			continue;
// 		}
		if (_fp->particleType() == DUMMY){
			float bp = fp[_fp->baseFluid()].pressure();
			/*_fp->setPressure(bp + _fp->ghostPressure());*/
			_fp->setPressure(bp + _fp->ghostPressure());
		}
// 		float bp = src[_fp->baseFluid()];
// 		/*_fp->setPressure(bp + _fp->ghostPressure());*/
// 		src[j] = bp + _fp->ghostPressure();
	}
}

void incompressible_sph::dummyScalarCopy(float* src)
{
	fluid_particle *_fp = NULL;
	for (size_t j = particleCountByType[FLUID]; j < np; j++){
	//for (size_t j = 0; j < np; j++){
		_fp = fp + j;
		if (_fp->particleType() != BOUNDARY)
			continue;
		float vec = src[j];
		size_t k = j + 1;
		while (k < np && fp[k].particleType() == DUMMY)
			src[k++] = vec;// +fp[k].hydroPressure();
	}
}

float incompressible_sph::dotProductIgnoreType(float* v1, float* v2, tParticle tp)
{
	float out = 0.f;
	for (size_t j = 0; j < np; j++){
		if (fp[j].particleType() != tp){
			out += v1[j] * v2[j];
		}
	}
	return out;
}

bool incompressible_sph::solvePressureWithBiCGSTAB()
{
	float* lhs = new float[np];
	//float* rhs = new float[np];
	float* conjugate0 = new float[np];
	float* conjugate1 = new float[np];
	float* tmp0 = new float[np];
	float* tmp1 = new float[np];
	float* residual = new float[np];
	

	PPESolver(lhs);
	float ip_rr = 0.f;
	fluid_particle *_fp = NULL;
	float lhs78 = lhs[78];
	for (size_t i = 0; i < np; i++){
		conjugate0[i] = 0.f;
		conjugate1[i] = 0.f;
		tmp0[i] = 0.f;
		tmp1[i] = 0.f;
		residual[i] = 0.f;
		conjugate0[i] = residual[i] = rhs[i] = rhs[i] - lhs[i];
		if (fp[i].particleType() == FLUID)
			ip_rr += residual[i] * residual[i];
	}
	//float lhs78 = lhs[78];
	if (ip_rr <= DBL_EPSILON)
	{
		std::cout << "parameter 'ip_rr' is wrong value. - PPESolver_CPU_VERSION - " << std::endl;
		return false;
	}
	float norm_sph_sqared = ip_rr;
	float residual_norm_squared;
	float alpha = 0.f;
	float omega = 0.f;
	float beta = 0.f;
	float malpha = 0.f;
	float dot1 = 0.f;
	float dot2 = 0.f;
	//const size_t c_np = np;
	for (size_t i = 0; i < ppeIter; i++){
// 		if (boundaryTreatment() == GHOST_PARTICLE_METHOD)
// 			ghostDummyScalarSet(conjugate0);
// 		else
// 			dummyScalarCopy(conjugate0);
		
		PPESolver(tmp0, conjugate0);
		alpha = ip_rr / dotProductIgnoreType(rhs, tmp0, FLUID);
		malpha = -alpha;
		for (size_t j = 0; j < np; j++)
			conjugate1[j] = malpha * tmp0[j] + residual[j];
// 		if (boundaryTreatment() == GHOST_PARTICLE_METHOD)
// 			ghostDummyScalarSet(conjugate1);
// 		else
// 			dummyScalarCopy(conjugate1);
		PPESolver(tmp1, conjugate1);
		omega = dotProductIgnoreType(tmp1, conjugate1, FLUID) / dotProductIgnoreType(tmp1, tmp1, FLUID);
		for (size_t j = 0; j < np; j++){
			if (fp[j].particleType() == FLUID)
			{
				// continue;
				float _pes = fp[j].pressure() + (alpha * conjugate0[j] + omega * conjugate1[j]);
				fp[j].setPressure(_pes);
				residual[j] = conjugate1[j] - omega * tmp1[j];
			}
		}
		residual_norm_squared = dotProductIgnoreType(residual, residual, FLUID);
		if (abs(residual_norm_squared / norm_sph_sqared) <= ppeTol * ppeTol){
			std::cout << "niteration : " << i << std::endl;
			break;
		}
		float new_ip_rr = dotProductIgnoreType(residual, rhs, FLUID);
		beta = (new_ip_rr / ip_rr) * (alpha / omega);
		ip_rr = new_ip_rr;
		for (size_t j = 0; j < np; j++){
			conjugate0[j] = residual[j] + beta*(conjugate0[j] - omega * tmp0[j]);
		}
	}
	if (boundaryTreatment() == DUMMY_PARTICLE_METHOD)
	{
		for (size_t i = 0; i < np; i++){
// 			if (fp[i].IsFreeSurface() && fp[i].particleType() == FLUID){
// 				fp[i].setPressure(0.f);
// 				continue;
// 			}
			if (fp[i].particleType() == BOUNDARY){
// 				if (fp[i].IsFreeSurface())
// 					fp[i].setPressure(0.f);
				float press = fp[i].pressure();
				size_t j = i + 1;
				while (j < np && fp[j].particleType() == DUMMY){
					fp[j].setPressure(press);
					j++;
				}
			}
		}
	}
	else{
		fluid_particle *_fp = NULL;
		for (size_t j = 0; j < np; j++){
			_fp = fp + j;
// 			if (_fp->IsFreeSurface()/* && _fp->particleType()!=DUMMY*/){
// 				_fp->setPressure(0.f);
// 				continue;
// 			}
			if (_fp->particleType() == DUMMY){
				float bp = fp[_fp->baseFluid()].pressure();
				_fp->setPressure(bp + _fp->ghostPressure());
				//_fp->setPressure(bp/* + _fp->ghostPressure()*/);
			}
		}
	}

// 	for (size_t i = 0; i < np; i++){
// 		if (fp[i].IsFreeSurface() && fp[i].particleType()==FLUID)
// 		{
// 			fp[i].setPressure(0.f);
// 		}
// 	}
// 	for (size_t i = 0; i < np; i++){
// 		if (fp[i].particleType() == BOUNDARY){
// // 			if (fp[i].IsFreeSurface()){
// // 				fp[i].setPressure(0.f);
// // 				continue;
// // 			}
// 			float press = fp[i].pressure();
// 			size_t j = i + 1;
// 			while (j < np && fp[j].particleType() == DUMMY){
// 				fp[j].setPressure(press);
// 				j++;
// 			}
// 		}
// 	}
// 	for (size_t i = 0; i < np; i++){
// 		_fp = fp + i;
// 		if (_fp->IsFreeSurface())
// 			_fp->setPressure(0.f);
// 		if (_fp->particleType() == BOUNDARY){
// 			float _pes = _fp->pressure();
// 			size_t j = i + 1;
// 			while (j < np && _fp->particleType() == DUMMY){
// 				_fp->setPressure(_pes + fp[j].hydroPressure());
// 				j++;
// 			}
// 		}
// 	}

	delete[] lhs;// float* lhs = new float[np];
//	delete[] rhs;// float* rhs = new float[np];
	delete[] conjugate0;// float* conjugate0 = new float[np];
	delete[] conjugate1;// float* conjugate1 = new float[np];
	delete[] tmp0;// float* tmp0 = new float[np];
	delete[] tmp1;// float* tmp1 = new float[np];
	delete[] residual;// float* residual = new float[np];
	return true;
}

void incompressible_sph::PPESolver(float *out, float *pes)
{
	fluid_particle *_fp = NULL;
	VEC3F ip, dp;
	float ipress = 0.f;
	float dpress = 0.f;
	float _press = 0.f;
	float press = 0.f;
	size_t nnb = 0;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
	//	nnb = 0;
		press = 0.f;
		if (_fp->particleType() == DUMMY){
			out[i] = 0.f;
			continue;
		}
		ip = _fp->auxPosition();
		ipress = pes ? pes[i] : _fp->pressure();
		if (_fp->particleType() == FLUID)
			if (_fp->IsFreeSurface())
				ipress *= 2;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (/*_fp->particleType() == BOUNDARY && */it->j->particleType() == BOUNDARY)
				continue;
			if (i != it->j->ID())
			{
				dp = ip - it->j->auxPosition();
				dpress = ipress - (pes ? pes[it->j->ID()] : it->j->pressure());
				_press = it->j->mass() * dpress * dp.dot(it->gradW) / (dp.dot() + depsilon);
				press += _press;
			}
		}
		press *= 2.f / rho;
		out[i] = press;
	}
}

void incompressible_sph::correctionStep()
{
	fluid_particle *_fp = NULL;
	float pi, pj;
	VEC3F gradp, pd, acci, nv, pos, vel;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;

		gradp = 0.f;
		if (_fp->particleType() != FLUID)
			return;
		pi = _fp->pressure() / (rho * rho);
		pj = 0.f;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (it->j->particleType() == BOUNDARY)
				continue;
			pd = _fp->auxPosition() - it->j->auxPosition();
			pj = it->j->pressure() / (rho * rho);
// 			if (it->j->IsFreeSurface())
// 				gradp += _fp->mass() * 1.5f * pi * it->gradW;
// 			else
// 			if (_fp->IsFreeSurface())
// 				gradp += _fp->mass() * pj * it->gradW;
// 			else if (it->j->IsFreeSurface())
// 				gradp += _fp->mass() * pi * it->gradW;
// 			else
				gradp += it->j->mass() *(pi + pj) * it->gradW;			
		}
		acci = gradp;
		nv = _fp->auxVelocity() - dt * acci;
		pos = _fp->auxPosition() + 0.5f * dt * (nv + _fp->velocity());
		vel = nv;
		if (i == 7140)
			std::cout << vel << std::endl;
		_fp->setPositionTemp(pos);
		_fp->setVelocityTemp(vel);
		if (vel.length() > maxVel)
			maxVel = vel.length();
		if (acci.length() > maxAcc)
			maxAcc = acci.length();
	}
}

void incompressible_sph::calc_viscous()
{
	//for (size_t i = 0; i < particleCountByType[FLUID]; i++){
	//fluid_particle *_fp = &fp[i];
	for (unsigned int i = 0; i < particleCountByType[FLUID]; i++){
		VEC3F rij, uij;
		float up, bot;
		VEC3F rst = 0.f;
		fluid_particle* _fp = &fp[i];
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (it->j->particleType() == BOUNDARY) continue;
			rij = _fp->position() - it->j->position();
			uij = _fp->velocity() - it->j->velocity();
			up = 4.f * it->j->mass() * (dynVisc + dynVisc) * rij.dot(it->gradW);
			bot = pow(rho + rho, 2.f) * (rij.dot() + depsilon);
			rst += (up / bot) * uij;
		}
		_fp->setViscousTerm(rst);
	}
	//}
}

void incompressible_sph::calc_sps_turbulence()
{
	float dp_sps = sqrt(pspace * pspace * 2.f) / 2.f;
	float sps_smag = pow((0.12f * dp_sps), 2.f);
	float sps_blin = (2.f / 3.f) * 0.0066f * dp_sps * dp_sps;
	for (unsigned int i = 0; i < particleCountByType[FLUID]; i++){
		VEC3F uij;
		symatrix gradVel = { 0, };
		float vol;
		float dv;
		fluid_particle* _fp = &fp[i];
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (it->j->particleType() == BOUNDARY) continue;
			uij = _fp->velocity() - it->j->velocity();
			vol = it->j->mass() / rho;
			dv = uij.x * vol; gradVel.xx += dv * it->gradW.x; gradVel.xy += dv * it->gradW.y; gradVel.xz += dv * it->gradW.z;
			dv = uij.y * vol; gradVel.xy += dv * it->gradW.x; gradVel.yy += dv * it->gradW.y; gradVel.yz += dv * it->gradW.z;
			dv = uij.z * vol; gradVel.xz += dv * it->gradW.x; gradVel.yz += dv * it->gradW.y; gradVel.zz += dv * it->gradW.z;
		}
		const float pow1 = gradVel.xx * gradVel.xx + gradVel.yy * gradVel.yy + gradVel.zz * gradVel.zz;
		const float prr = pow1 + pow1 + gradVel.xy * gradVel.xy + gradVel.xz * gradVel.xz + gradVel.yz * gradVel.yz;
		const float visc_sps = sps_smag * sqrt(prr);
		const float div_u = gradVel.xx + gradVel.yy + gradVel.zz;
		const float sps_k = (2.f / 3.f) * visc_sps * div_u;
		const float sps_bn = sps_blin * prr;
		const float sumsps = -(sps_k + sps_blin);
		const float twovisc_sps = (visc_sps + visc_sps);
		const float one_rho2 = 1.f / rho;
		symatrix tau;
		tau.xx = one_rho2 * (twovisc_sps * gradVel.xx + sumsps);
		tau.xy = one_rho2 * (visc_sps * gradVel.xy);
		tau.xz = one_rho2 * (visc_sps * gradVel.xz);
		tau.yy = one_rho2 * (twovisc_sps * gradVel.yy + sumsps);
		tau.yz = one_rho2 * (visc_sps * gradVel.yz);
		tau.zz = one_rho2 * (twovisc_sps * gradVel.zz + sumsps);
		_fp->setTau(tau);
	}
	
}

void incompressible_sph::first_step()
{
	calc_sps_turbulence();
	calc_viscous();
	for (size_t i = 0; i < particleCountByType[FLUID]; i++){
		fluid_particle *_fp = &fp[i];
		VEC3F visc = _fp->viscoustTerm();
		symatrix tau1 = _fp->tau();
		VEC3F eddy;
		float tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (it->j->particleType() == BOUNDARY) continue;
			symatrix tau2 = it->j->tau();
			tau_xx = tau1.xx + tau2.xx; tau_xy = tau1.xy + tau2.xy; tau_xz = tau1.xz + tau2.xz;
			tau_yy = tau1.yy + tau2.yy; tau_yz = tau1.yz + tau2.yz; tau_zz = tau1.zz + tau2.zz;
			eddy.x += it->j->mass() * (tau_xx * it->gradW.x + tau_xy * it->gradW.y + tau_xz * it->gradW.z);
			eddy.y += it->j->mass() * (tau_xy * it->gradW.x + tau_yy * it->gradW.y + tau_yz * it->gradW.z);
			eddy.z += it->j->mass() * (tau_xz * it->gradW.x + tau_yz * it->gradW.y + tau_zz * it->gradW.z);
		}
		_fp->setAuxVelocity(_fp->velocity() + dt * (visc/* + eddy*/));
		_fp->setAuxPosition(_fp->position() + dt * _fp->auxVelocity());
	}
}

void incompressible_sph::predictionStep2()
{
	fluid_particle *_fp = NULL;
	VEC3F ip, iv, dp, dv;
	VEC3I cell, loopStart, loopEnd;
	float div_u = 0.f;
	if (rhs)
		delete[] rhs;
	rhs = new float[np];
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		if (_fp->particleType() == DUMMY){
			rhs[i] = 0.f;
			continue;
		}

		div_u = 0.0f;
		ip = _fp->auxPosition();
		iv = _fp->auxVelocity();
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (it->j->particleType() == BOUNDARY)
				continue;
			dp = ip - it->j->auxPosition();
			dv = iv - it->j->auxVelocity();
			div_u += (it->j->mass() / rho) * dv.dot(it->gradW);
		}
		rhs[i] = -div_u * dt_inv;
	}
}

void incompressible_sph::PPESolver2(float *out, float *pes)
{
	fluid_particle *_fp = NULL;
	VEC3F ip, jp, rij;
	float ipress = 0.f;
	float jpress = 0.f;
	float dpress = 0.f;
	float _press = 0.f;
	float press = 0.f;
	size_t nnb = 0;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		//	nnb = 0;
		press = 0.f;
		if (_fp->particleType() == DUMMY){
			out[i] = 0.f;
			continue;
		}
		ip = _fp->auxPosition();
		ipress = pes ? pes[i] : _fp->pressure();
		if (_fp->particleType() == FLUID)
			if (_fp->IsFreeSurface())
				ipress *= 2;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (/*_fp->particleType() == BOUNDARY && */it->j->particleType() == BOUNDARY)
				continue;
			if (i != it->j->ID())
			{
				jpress = pes ? pes[it->j->ID()] : it->j->pressure();
				jp = it->j->auxPosition();
				dpress = ipress - jpress;
				rij = ip - jp;
				press += it->j->mass() * (dpress * rij.dot(it->gradW)) / (rij.dot() + depsilon);
			}
		}
		press *= 8.0f / pow(rho + rho, 2.f);
		out[i] = press;
	}
}

bool incompressible_sph::solvePressureWithBiCGSTAB2()
{
	float* lhs = new float[np];
	//float* rhs = new float[np];
	float* conjugate0 = new float[np];
	float* conjugate1 = new float[np];
	float* tmp0 = new float[np];
	float* tmp1 = new float[np];
	float* residual = new float[np];


	PPESolver2(lhs);
	float ip_rr = 0.f;
	fluid_particle *_fp = NULL;
	float lhs78 = lhs[78];
	for (size_t i = 0; i < np; i++){
		conjugate0[i] = 0.f;
		conjugate1[i] = 0.f;
		tmp0[i] = 0.f;
		tmp1[i] = 0.f;
		residual[i] = 0.f;
		conjugate0[i] = residual[i] = rhs[i] = rhs[i] - lhs[i];
		if (fp[i].particleType() == FLUID)
			ip_rr += residual[i] * residual[i];
	}
	//float lhs78 = lhs[78];
	if (ip_rr <= DBL_EPSILON)
	{
		std::cout << "parameter 'ip_rr' is wrong value. - PPESolver_CPU_VERSION - " << std::endl;
		return false;
	}
	float norm_sph_sqared = ip_rr;
	float residual_norm_squared;
	float alpha = 0.f;
	float omega = 0.f;
	float beta = 0.f;
	float malpha = 0.f;
	float dot1 = 0.f;
	float dot2 = 0.f;
	//const size_t c_np = np;
	for (size_t i = 0; i < ppeIter; i++){
		PPESolver2(tmp0, conjugate0);
		alpha = ip_rr / dotProductIgnoreType(rhs, tmp0, FLUID);
		malpha = -alpha;
		for (size_t j = 0; j < np; j++)
			conjugate1[j] = malpha * tmp0[j] + residual[j];
		PPESolver2(tmp1, conjugate1);
		omega = dotProductIgnoreType(tmp1, conjugate1, FLUID) / dotProductIgnoreType(tmp1, tmp1, FLUID);
		for (size_t j = 0; j < np; j++){
			if (fp[j].particleType() == FLUID)
			{
				// continue;
				float _pes = fp[j].pressure() + (alpha * conjugate0[j] + omega * conjugate1[j]);
				fp[j].setPressure(_pes);
				residual[j] = conjugate1[j] - omega * tmp1[j];
			}
		}
		residual_norm_squared = dotProductIgnoreType(residual, residual, FLUID);
		if (abs(residual_norm_squared / norm_sph_sqared) <= ppeTol * ppeTol){
			std::cout << "niteration : " << i << std::endl;
			break;
		}
		float new_ip_rr = dotProductIgnoreType(residual, rhs, FLUID);
		beta = (new_ip_rr / ip_rr) * (alpha / omega);
		ip_rr = new_ip_rr;
		for (size_t j = 0; j < np; j++){
			conjugate0[j] = residual[j] + beta*(conjugate0[j] - omega * tmp0[j]);
		}
	}
	if (boundaryTreatment() == DUMMY_PARTICLE_METHOD)
	{
		for (size_t i = 0; i < np; i++){
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
	else{
		fluid_particle *_fp = NULL;
		for (size_t j = 0; j < np; j++){
			_fp = fp + j;
			if (_fp->particleType() == FLUID && _fp->IsFreeSurface())
				_fp->setPressure(0.f);
			if (_fp->particleType() == DUMMY){
				float bp = fp[_fp->baseFluid()].pressure();
				_fp->setPressure(bp + _fp->ghostPressure());
			}
		}
	}
	delete[] lhs;// float* lhs = new float[np];
	//delete[] rhs;// float* rhs = new float[np];
	delete[] conjugate0;// float* conjugate0 = new float[np];
	delete[] conjugate1;// float* conjugate1 = new float[np];
	delete[] tmp0;// float* tmp0 = new float[np];
	delete[] tmp1;// float* tmp1 = new float[np];
	delete[] residual;// float* residual = new float[np];
	return true;
}

void incompressible_sph::correctionStep2()
{
	fluid_particle *_fp = NULL;
	float pi, pj;
	VEC3F gradp, pd, acci, nv, pos, vel;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		gradp = 0.f;
		if (_fp->particleType() != FLUID)
			return;
		pi = _fp->pressure() / (rho * rho);
		pj = 0.f;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (i == 3120)
				i = 3120;
			if (it->j->particleType() == BOUNDARY)
				continue;
			pd = _fp->auxPosition() - it->j->auxPosition();
			pj = it->j->pressure() / (rho * rho);
			gradp += it->j->mass() *(pi + pj) * it->gradW;
		}
		acci = grav - gradp;
		nv = _fp->auxVelocity() + dt * acci;
		pos = _fp->position() + 0.5f * dt * (nv + _fp->velocity());
		vel = nv;
		_fp->setPosition(pos);
		_fp->setVelocity(vel);
		if (vel.length() > maxVel)
			maxVel = vel.length();
		if (acci.length() > maxAcc)
			maxAcc = acci.length();
	}
}

void incompressible_sph::predict_the_acceleration()
{
	fluid_particle *_fp = NULL;
	VEC3F ip, iv, ia, dp, dv;
	float p_1 = 0, p_2 = 0;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		if (_fp->particleType() == BOUNDARY || _fp->particleType() == DUMMY)
			continue;
		ip = _fp->position();
		iv = _fp->velocity();
		ia = grav;

		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
// 			if (it->j->particleType() == BOUNDARY)
// 				continue;
			dp = ip - it->j->position();
			dv = iv - it->j->velocity();
			p_1 = 8 * (dynVisc + dynVisc) / (rho + rho);
			p_2 = dv.dot(dp) / (dp.dot(dp) + depsilon);
			ia += it->j->mass() * (p_1 * p_2) * it->gradW;
		}
		_fp->setAcceleration(ia);
		//_fp->setAuxVelocity(iv + dt * ia);
	}
}

void incompressible_sph::predict_the_temporal_position()
{
	fluid_particle *_fp = NULL;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		if (_fp->particleType() != FLUID)
			continue;
		_fp->setAuxPosition(_fp->position());
		_fp->setPosition(_fp->position() + dt * _fp->velocity());
	}
}

void incompressible_sph::predict_the_temporal_velocity()
{
	fluid_particle *_fp = NULL;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		if (_fp->particleType() == BOUNDARY || _fp->particleType() == DUMMY)
			continue;
		_fp->setAuxVelocity(_fp->velocity() + dt * _fp->acceleration());
// 		_fp->setAuxPosition(_fp->position());
// 		_fp->setPosition(_fp->position() + dt * _fp->auxVelocity());
	}
}

void incompressible_sph::pressure_poisson_equation(float* out, float* p)
{
	fluid_particle *_fp = NULL;
	VEC3F ip, jp, rij;
	float ipress = 0.f;
	float jpress = 0.f;
	float dpress = 0.f;
	float _press = 0.f;
	float press = 0.f;
	size_t nnb = 0;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		press = 0.f;
		if (_fp->particleType() == DUMMY){
			out[i] = 0.f;
			continue;
		}
		ip = _fp->position();
		ipress = p ? p[i] : _fp->pressure();
		if (_fp->particleType() == FLUID || _fp->particleType() == FLOATING)
			if (_fp->IsFreeSurface())
				ipress *= 1.8f;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			if (i != it->j->ID())
			{
				jpress = p ? p[it->j->ID()] : it->j->pressure();
				jp = it->j->position();
				dpress = ipress - jpress;
				rij = ip - jp;
				float mp = it->j->mass() * (dpress * rij.dot(it->gradW)) / (rij.dot() + depsilon);
				if (it->j->ID() == 152)
					mp = mp;
				press += mp;// it->j->mass() * (dpress * rij.dot(it->gradW)) / (rij.dot() + depsilon);
			}
		}
		press *= 2.0f / rho;
		out[i] = press;
	}
}

void incompressible_sph::ppe_right_hand_side(float* out)
{
	fluid_particle *_fp = NULL;
	VEC3F ip, iv, dp, dv;
	VEC3I cell, loopStart, loopEnd;
	float div_u = 0.f;
// 	if (rhs)
// 		delete[] rhs;
// 	rhs = new float[np];
// 	memset(rhs, 0, sizeof(float) * np);
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		if (_fp->particleType() == DUMMY){
			out[i] = 0.f;
			continue;
		}

		div_u = 0.0f;
		iv = _fp->auxVelocity();
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			dv = iv - it->j->auxVelocity();
			div_u += it->j->mass() * dv.dot(it->gradW);
		}
		div_u *= -(1 / rho);
		out[i] = (rho / dt) * div_u;
	}
}

size_t incompressible_sph::solve_the_pressure_poisson_equation_by_Bi_CGSTAB()
{
	float* lhs = new float[np];
	float* _rhs = new float[np];
	float* conjugate0 = new float[np];
	float* conjugate1 = new float[np];
	float* tmp0 = new float[np];
	float* tmp1 = new float[np];
	float* residual = new float[np];
	ppe_right_hand_side(_rhs);
	pressure_poisson_equation(lhs);
	float ip_rr = 0.f;
	fluid_particle *_fp = NULL;
	float lhs78 = lhs[78];
	for (size_t i = 0; i < np; i++){
		conjugate0[i] = 0.f;
		conjugate1[i] = 0.f;
		tmp0[i] = 0.f;
		tmp1[i] = 0.f;
		residual[i] = 0.f;
		conjugate0[i] = residual[i] = _rhs[i] = _rhs[i] - lhs[i];
		if (fp[i].particleType() != DUMMY)
			ip_rr += residual[i] * residual[i];
	}
	//float lhs78 = lhs[78];
	if (ip_rr <= DBL_EPSILON)
	{
		std::cout << "parameter 'ip_rr' is wrong value. - PPESolver_CPU_VERSION - " << std::endl;
		return false;
	}
	float norm_sph_sqared = ip_rr;
	float residual_norm_squared;
	float alpha = 0.f;
	float omega = 0.f;
	float beta = 0.f;
	float malpha = 0.f;
	float dot1 = 0.f;
	float dot2 = 0.f;
	size_t it = 0;
	for (it = 0; it < ppeIter; it++){
// 		for (size_t i = 0; i < np; i++){
// 			if (fp[i].particleType() == BOUNDARY){
// 				float press = conjugate0[i];
// 				size_t j = i + 1;
// 				while (j < np && fp[j].particleType() == DUMMY){
// 					conjugate0[j] = press;
// 					j++;
// 				}
// 			}
// 		}
		pressure_poisson_equation(tmp0, conjugate0);
// 		for (size_t i = 0; i < np; i++){
// 			if (fp[i].particleType() == BOUNDARY){
// 				float press = tmp0[i];
// 				size_t j = i + 1;
// 				while (j < np && fp[j].particleType() == DUMMY){
// 					tmp0[j];
// 					j++;
// 				}
// 			}
// 		}
		alpha = ip_rr / dotProductIgnoreType(_rhs, tmp0, DUMMY);
		malpha = -alpha;
		for (size_t j = 0; j < np; j++)
			conjugate1[j] = malpha * tmp0[j] + residual[j];

// 		for (size_t i = 0; i < np; i++){
// 			if (fp[i].particleType() == BOUNDARY){
// 				float press = conjugate1[i];
// 				size_t j = i + 1;
// 				while (j < np && fp[j].particleType() == DUMMY){
// 					conjugate1[j] = press;
// 					j++;
// 				}
// 			}
// 		}

		pressure_poisson_equation(tmp1, conjugate1);
	
		omega = dotProductIgnoreType(tmp1, conjugate1, DUMMY) / dotProductIgnoreType(tmp1, tmp1, DUMMY);
		for (size_t j = 0; j < np; j++){
			if (fp[j].particleType() != DUMMY)
			{
				float _pes = fp[j].pressure() + (alpha * conjugate0[j] + omega * conjugate1[j]);
				fp[j].setPressure(_pes);
				residual[j] = conjugate1[j] - omega * tmp1[j];
			}
		}
		residual_norm_squared = dotProductIgnoreType(residual, residual, DUMMY);
		if (abs(residual_norm_squared / norm_sph_sqared) <= ppeTol * ppeTol){
			//std::cout << "niteration : " << i << std::endl;
			break;
		}
		float new_ip_rr = dotProductIgnoreType(residual, _rhs, DUMMY);
		beta = (new_ip_rr / ip_rr) * (alpha / omega);
		ip_rr = new_ip_rr;
		for (size_t j = 0; j < np; j++){
			conjugate0[j] = residual[j] + beta*(conjugate0[j] - omega * tmp0[j]);
		}
	}
	if (boundaryTreatment() == DUMMY_PARTICLE_METHOD)
	{
		for (size_t i = 0; i < np; i++){
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

	delete[] lhs;// float* lhs = new float[np];
	delete[] _rhs;// float* _rhs = new float[np];
	delete[] conjugate0;// float* conjugate0 = new float[np];
	delete[] conjugate1;// float* conjugate1 = new float[np];
	delete[] tmp0;// float* tmp0 = new float[np];
	delete[] tmp1;// float* tmp1 = new float[np];
	delete[] residual;// float* residual = new float[np];

	return it;
}

void incompressible_sph::correct_by_adding_the_pressure_gradient_term()
{
	fluid_particle *_fp = NULL;
	maxVel = 0.f;
	float pi, pj, pij;
	VEC3F gradp, acci, nv, pos, vel;
	for (size_t i = 0; i < np; i++){
		_fp = fp + i;
		gradp = 0.f;
		if (_fp->particleType() == BOUNDARY || _fp->particleType() == DUMMY)
			return;
		pi = _fp->pressure();
		pj = 0.f;
		for (NeighborIterator it = _fp->BeginNeighbor(); it != _fp->EndNeighbor(); it++){
			pj = it->j->pressure();
			pij = (pi + pj) / (rho * rho);
// 			if (it->j->IsFreeSurface())
// 				gradp += 2 * it->j->mass() * (pi / (rho * rho)) * it->gradW;
// 			else
			gradp += it->j->mass() * pij * it->gradW;
		}
		acci = gradp;
		nv = _fp->auxVelocity() - dt * acci;
	//	nv = _fp->auxVelocity() + dt*(grav - acci);
		pos = _fp->position() + dt * nv;
	//	pos = _fp->position() + dt * 0.5f * (_fp->velocity() + nv);
		vel = nv;
// 		if (i == 7140)
// 			std::cout << vel << std::endl;
		if(_fp->particleType() == FLUID)
			_fp->setPosition(pos);
		_fp->setVelocity(vel);
// 		float ace = (_fp->acceleration() - acci).length();
// 		if (maxAcc < ace){
// 			maxAcc = ace;
// 		}
		float v = vel.length();
		if (v > maxVel)
			maxVel = v;
// 		if (acci.length() > maxAcc)
// 			maxAcc = acci.length();
	}
}

void incompressible_sph::particle_shifting()
{
	for (unsigned int i = 0; i < np; i++){
		fluid_particle *parI = particle(i);
		parI->setPositionTemp(parI->position());
		parI->setPressureTemp(parI->pressure());
		parI->setVelocityTemp(parI->velocity());
	}
	float A_fsm = 2.f;
	float A_fst = 1.5f;
	float A_fsc = 0.f;
	VEC3F dr;
	for (unsigned int i = 0; i < np; i++){
		fluid_particle *parI = particle(i);
		if (parI->particleType() != FLUID) continue;
		VEC3F gradC;
		for (NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			gradC += (it->j->mass() / it->j->density()) * it->gradW;
		}
		float v_mag = parI->velocity().length();
		float div_r = parI->divR();
		A_fsc = (div_r - A_fst) / (A_fsm - A_fst);
		if (div_r < A_fst)
			dr = -A_fsc * 2.0f * skernel.h * v_mag * dt * gradC;
		else
			dr = -2.f * skernel.h * v_mag * dt * gradC;
		parI->setPosition(parI->position() + dr);
	}

// 	VEC3F posDif;
// 	for (unsigned int i = 0; i < np; i++){
// 		fluid_particle *parI = particle(i);
// 		tParticle type = parI->particleType();
// 		if (type != FLUID || parI->IsFreeSurface()){
// 			continue;
// 		}
// 
// 		VEC3F posI = parI->positionTemp();
// 		float effRadiusSq = sphkernel->KernelSupprotSq() * skernel.h_sq;
// 
// 		for (NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
// 			size_t j = it->j->ID();
// 			posDif = posI - it->j->positionTemp();
// 			if (it->j->IsFreeSurface()){
// 				float distSq = posDif.dot();
// 				if (distSq < effRadiusSq)
// 					effRadiusSq = distSq;
// 			}
// 		}
// 
// 		int neighborCount = 0;
// 		float avgSpacing = 0;
// 		VEC3F shiftVec = VEC3F(0.0);
// 
// 		for (NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
// 			size_t j = it->j->ID();
// 			posDif = posI - it->j->positionTemp();
// 			float distSq = posDif.dot();
// 			if (distSq > effRadiusSq)
// 				continue;
// 			float dist = sqrt(distSq);
// 			neighborCount++;
// 			avgSpacing += dist;
// 			shiftVec += posDif / (distSq * dist);
// 		}
// 
// 		if (!neighborCount)
// 			continue;
// 
// 		avgSpacing /= neighborCount;
// 		shiftVec *= avgSpacing * avgSpacing;
// 
// 		float velMagnitude = parI->velocity().length();
// 		shiftVec = min(shiftVec.length() * pshift.factor * velMagnitude * dt, pspace) * shiftVec.normalize();
// 		parI->setPosition(parI->position() + shiftVec);
//	}
}

void incompressible_sph::update_shift_particle()
{
	for (size_t i = 0; i < np; i++){
		fluid_particle *parI = particle(i);
		tParticle type = parI->particleType();
		if (type != FLUID || parI->IsFreeSurface())
			continue;

		VEC3F posI = parI->positionTemp();
		VEC3F gp = VEC3F(0.f);
		VEC3F gvx = VEC3F(0.f);
		VEC3F gvy = VEC3F(0.f);
		VEC3F velI = parI->velocityTemp();
		float pI = parI->pressureTemp();

		for (NeighborIterator it = parI->BeginNeighbor(); it != parI->EndNeighbor(); it++){
			size_t j = it->j->ID();
			VEC3F velJ = it->j->velocityTemp();
			VEC3F gradW = (it->j->mass() / it->j->density()) * it->gradW;
			gp += (it->j->pressureTemp() + pI) * gradW;
			gvx += (velJ.x - velI.x) * gradW;
			gvy += (velJ.y - velI.y) * gradW;
		}
		VEC3F dr = parI->position() - posI;
		parI->setPressure(parI->pressure() + gp.dot(dr));// += gp.dot(dr);
		parI->setVelocity(VEC3F(gvx.dot(dr), gvy.dot(dr), 0.0));// += vector3<double>(gvx.dot(dr), gvy.dot(dr), 0.0);
	}
}

void incompressible_sph::update_floating_body()
{
	size_t nfloat = particleCountByType[FLOATING];
	size_t init = particleCountByType[FLUID];
	size_t endt = init + particleCountByType[FLOATING];
	VEC3F rc = 0.f;
	for (size_t i = init; i < endt; i++){
		rc += fp[i].position();
	}
	rc = rc / (float)nfloat;
	float inertia = 0.f;
	VEC3F T = 0.f;
	VEC3F R = 0.f;
	for (size_t i = init; i < endt; i++){
		VEC3F qk = fp[i].position() - rc;
		inertia += qk.length() * qk.length();
		T += fp[i].velocity();
		R += fp[i].velocity().cross(qk);
	}
	T = T / (float)nfloat;
	R = R / inertia;
	for (size_t i = init; i < endt; i++){
		VEC3F qk = fp[i].position() - rc;
		VEC3F new_v = T + qk.cross(R);
		fp[i].setVelocity(new_v);
		VEC3F new_p = fp[i].position() + dt * new_v;
		fp[i].setPosition(new_p);
	}
}

void incompressible_sph::cpuRun()
{
	size_t part = 0;
	size_t cstep = 0;
	size_t eachStep = 0;
	size_t ppe_iter = 0;
	size_t nstep = static_cast<size_t>((et / dt) + 1);
	float ct = dt * cstep;
	parSIM::timer tmer;
	std::cout << "----------------------------------------------------------------------" << std::endl
			  << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time | I. ppe |" << std::endl
			  << "----------------------------------------------------------------------" << std::endl;
	std::ios::right;
	std::setprecision(6);
	if (exportData(part++))
	{
		std::cout << "| " << std::setw(9) << part - 1 << std::setw(12) << ct << std::setw(10) << eachStep << std::setw(11) << cstep << std::setw(15) << 0 << std::setw(8) << ppe_iter << std::setw(0) << " |" << std::endl;
	}
	exportParticlePosition();
	tmer.Start();
	while (cstep < nstep){
		cstep++;
		eachStep++;
		ct = dt * cstep;
		runModelExpression(dt, ct);
		//predict_the_temporal_position();
		fd->sort();
		//gradientCorrection();
		predict_the_acceleration();
		predict_the_temporal_velocity();
	//	fd->sort();
		calcFreeSurface(false);
		ppe_iter += solve_the_pressure_poisson_equation_by_Bi_CGSTAB();
		
		correct_by_adding_the_pressure_gradient_term();
		if(particleCountByType[FLOATING])
			update_floating_body();
		////fd->sort(true);
		//first_step();
		////exportParticlePosition();
		//fd->sort(false);
		//calcFreeSurface(false);
		//predictionStep2();
		//if (!solvePressureWithBiCGSTAB2())
		//	return;
		//correctionStep2();
		if (pshift.enable){
			fd->sort();
			particle_shifting();
			//update_shift_particle();
		}
		//updateTimeStep();
		if (!((cstep) % sphydrodynamics::st)){
			tmer.Stop();
			if (exportData(part++)){
				std::cout << "| " << std::setw(9) << part - 1 << std::setw(12) << std::fixed << ct << std::setw(10) << eachStep << std::setw(11) << cstep << std::setw(15) << tmer.GetElapsedTimeF() << std::setw(8) << ppe_iter << std::setw(0) << " |" << std::endl;
			}
			ppe_iter = 0;
			eachStep = 0;
			tmer.Start();
		}
		//cstep++;
		//eachStep++;
		//ct = cstep * dt;
	}
}


// 
// void incompressible_sph::cpuRun()
// {
// 	if (fs.is_open())
// 		fs.close();
// 	float dur_t = 0.f;
// 	float part_t = 0.f;
// 	size_t part = 0;
// //	exportData(part++);
// 	while (et > dur_t){
// 		std::cout << dur_t << std::endl;
// // 		if (part_t > st){
// // 			exportData(part++);
// // 		}
// 		//runModelExpression(dt, dur_t);
// 		auxiliaryPosition();
// 		fd->sort();
// 		//exportParticlePosition();
// 		calcFreeSurface(false);
// 		if (tCorr == GRADIENT_CORRECTION)
// 			gradientCorrection();
// 	
// 		auxiliaryVelocity();
// 
// 		predictionStep();
// 		if (!solvePressureWithBiCGSTAB())
// 			return;
// 		correctionStep();
// // 
// // 		dt = newTimeStep();
// // 		std::cout << "new timestep : " << dt << std::endl;
// 		dur_t += dt;
// 		part_t += dt;
// 		//if (part_t > st){
// 		//	exportData(part++);
// 	//	}
// 	}	
// }

void incompressible_sph::gpuRun()
{

}