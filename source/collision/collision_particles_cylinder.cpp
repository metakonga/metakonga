#include "collision_particles_cylinder.h"
#include "particle_system.h"
#include "cylinder.h"
#include "mphysics_cuda_dec.cuh"
#include "mass.h"

collision_particles_cylinder::collision_particles_cylinder()
{

}

collision_particles_cylinder::collision_particles_cylinder(QString& _name, modeler* _md, particle_system *_ps, cylinder *_cy)
	: collision(_name, _md, _ps->name(), _cy->objectName(), PARTICLES_CYLINDER)
	, ps(_ps)
	, cy(_cy)
{

}

collision_particles_cylinder::~collision_particles_cylinder()
{

}

bool collision_particles_cylinder::collid(float dt)
{
	
	return true;
}

bool collision_particles_cylinder::cuCollid()
{
	double3 *mforce;
	double3 *mmoment;
	double3 *mpos;
	VEC3D _mp;
	double3 _mf = make_double3(0.0, 0.0, 0.0);
	double3 _mm = make_double3(0.0, 0.0, 0.0);
	if(cy->pointMass())
		_mp = cy->pointMass()->getPosition();
	checkCudaErrors(cudaMalloc((void**)&mpos, sizeof(double3)));
	checkCudaErrors(cudaMemcpy(mpos, &_mp, sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&mforce, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMalloc((void**)&mmoment, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMemset(mforce, 0, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMemset(mmoment, 0, sizeof(double3)*ps->numParticle()));
	
	cu_cylinder_hertzian_contact_force(cy->deviceCylinderInfo(), cy->youngs(), cy->poisson(), rest, sratio, fric, ps->cuPosition(), ps->cuVelocity(), ps->cuOmega(), ps->cuForce(), ps->cuMoment(), ps->cuMass(), ps->youngs(), ps->poisson(), ps->numParticle(), mpos, mforce, mmoment, _mf, _mm);
	//std::cout << "_mf 1 :" << _mf.x << " " << _mf.y << " " << _mf.z << std::endl;
	_mf = reductionD3(mforce, ps->numParticle());
	//std::cout << "_mf 2 :" << _mf.x << " " << _mf.y << " " << _mf.z << std::endl;
	if (cy->pointMass()){
		cy->pointMass()->addCollisionForce(VEC3D(_mf.x, _mf.y, _mf.z));
	}
	/*std::cout << "_mm 1 :" << _mm.x << " " << _mm.y << " " << _mm.z << std::endl;*/
	_mm = reductionD3(mmoment, ps->numParticle());
	//std::cout << "_mm 2 :" << _mm.x << " " << _mm.y << " " << _mm.z << std::endl;
	if (cy->pointMass()){
		cy->pointMass()->addCollisionMoment(VEC3D(_mm.x, _mm.y, _mm.z));
	}
	//std::cout << h_mf->x << " " << h_mf->y << " " << h_mf->z << std::endl;
	checkCudaErrors(cudaFree(mforce)); mforce = NULL;
	checkCudaErrors(cudaFree(mmoment)); mmoment = NULL;
	checkCudaErrors(cudaFree(mpos)); mpos = NULL;
// 	delete h_mf; h_mf = NULL;
// 	delete h_mm; h_mm = NULL;
	return true;
}

float collision_particles_cylinder::particle_cylinder_contact_detection(VEC4F& pt, VEC3F& u, VEC3F& cp, unsigned int i)
{
// 	float dist = -1.0f;
// 	VEC3F loc = cy->origin();
// 	VEC3F xp = VEC3F(pt.x, pt.y, pt.z);
// 	VEC3F p1 = cy->basePos();
// 	VEC3F p2 = cy->topPos();
// 	VEC3F ab = p2 - p1;
// 	float t = (xp - p1).dot(ab) / ab.dot();
// 	if (t < 0.0f || t > 1.0f){
// 		cp = p1 + t * ab;
// 		dist = (xp - cp).length();
// 		if (dist < cy->topRadius()){
// 			float overlap = (cp - loc).length();
// 			u = (loc - cp) / overlap;
// 			cp = cp - cy->baseRadisu() * u;
// 			return cy->length() * 0.5f + pt.w - overlap;
// 		}
// 		VEC3F cyp;
// 		MAT33F A = cy->t_orientation().A();
// 		VEC3F _at = xp - cy->topPos();
// 		VEC3F at = transpose(A, _at);
// 		if (at.y < 0){
// 			A = cy->b_orientation().A();
// 			_at = xp - cy->basePos();
// 			at = transpose(A, _at);
// 		}
// 		float _r = _at.length();
// 		float r = at.length();		
// 		float th = acos(at.y / r);
// 		float pi = atan(at.x / at.z);
// 		cp.zeros();
// 		if (pi < 0 && at.z < 0){
// 			cp.x = cy->baseRadisu() * sin(-pi);
// 		}
// 		else if (pi > 0 && at.x < 0 && at.z < 0){
// 			cp.x = cy->baseRadisu() * sin(-pi);
// 		}
// 		else{
// 			cp.x = cy->baseRadisu() * sin(pi);
// 		}
// 		cp.z = cy->baseRadisu() * cos(pi);
// 		if (at.z < 0 && cp.z > 0){
// 			cp.z = -cp.z;
// 		}
// 		else if (at.z > 0 && cp.z < 0){
// 			cp.z = -cp.z;
// 		}
// 		cp.y = 0.f;
// 		VEC3F _cp;
// 		_cp += cy->basePos();
// 		float _dist = (cp - xp).length();
// 		cp = cy->topPos() + A * cp;
// 		VEC3F disVec = cp - xp;
// 		dist = disVec.length();
// 		u = disVec / dist;
// 		if (dist < pt.w){
// 			return pt.w - dist;
// 		}
// 		return -1.0f;
// 	}
// 	cp = p1 + t * ab;
// 	dist = (xp - cp).length();
// 	u = (cp - xp) / dist;
// 	cp = cp - cy->baseRadisu() * u;
// 	return cy->topRadius() + pt.w - dist;
	return 0.f;
}

bool collision_particles_cylinder::collid_with_particle(unsigned int i, float dt)
{
	switch (tcm)
	{
	case HMCM:
		this->HMCModel(i, dt);
		break;
	}
	return true;
}

bool collision_particles_cylinder::HMCModel(unsigned int i, float dt)
{
	VEC4F p = ps->position()[i];
	VEC3F v = ps->velocity()[i];
	VEC3F w = ps->angVelocity()[i];
	VEC3F u, Fn, Ft, M;
	VEC3F sF;
	VEC3D mforce, mmoment;
	VEC3F cp;
	float overlap = particle_cylinder_contact_detection(p, u, cp, i);
	VEC3D si = cp.To<double>() - cy->pointMass()->getPosition();
	//float overlap = (cy->topRadius() + p.w) - dist;
	if (overlap > 0)
	{
		VEC3F dv = -(v + w.cross(p.w * u));
		if (!cy->relativeImpactVelocity()[i])
			cy->relativeImpactVelocity()[i] = abs(dv.length());
		constant c = getConstant(p.w, 0.f, ps->mass()[i], 0.f, ps->youngs(), cy->youngs(), ps->poisson(), cy->poisson(),0.f);
		float fsn = (-c.kn * pow(overlap, 1.5f));
		float fca = cohesionForce(p.w, 0.f, ps->youngs(), 0.f, ps->poisson(), 0.f, fsn);
		Fn = (fsn + fca + c.vn * dv.dot(u)) * u;
		VEC3F e = dv - dv.dot(u) * u;
		float mag_e = e.length();
		VEC3F shf;
		if (mag_e){
			VEC3F s_hat = e / mag_e;
			float ds = mag_e * dt;
			Ft = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * Fn.length()) * s_hat;
			M = (p.w * u).cross(Ft);
		}
		sF = Fn + Ft;
		ps->force()[i] += sF;
		ps->moment()[i] += M;
		mforce = -sF.To<double>();
		mmoment = -si.cross(sF.To<double>());
	}
	else{
		cy->relativeImpactVelocity()[i] = 0.f;
	}
	if (cy->pointMass()){
		cy->pointMass()->addExternalForce(mforce);
		cy->pointMass()->addExternalMoment(mmoment);
	}
	
	return true;
}