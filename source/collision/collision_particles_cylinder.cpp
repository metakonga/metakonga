#include "collision_particles_cylinder.h"
#include "particle_system.h"
#include "cylinder.h"
#include "mass.h"

collision_particles_cylinder::collision_particles_cylinder()
{

}

collision_particles_cylinder::collision_particles_cylinder(
	QString& _name,
	modeler* _md, 
	particle_system *_ps,
	cylinder *_cy,
	tContactModel _tcm)
	: collision(_name, _md, _ps->name(), _cy->objectName(), PARTICLES_CYLINDER, _tcm)
	, ps(_ps)
	, cy(_cy)
{

}

collision_particles_cylinder::~collision_particles_cylinder()
{

}

bool collision_particles_cylinder::collid(double dt)
{
	
	return true;
}

bool collision_particles_cylinder::cuCollid(
	double *dpos, double *dvel,
	double *domega, double *dmass,
	double *dforce, double *dmoment, unsigned int np)
{
	double3 *mforce;
	double3 *mmoment;
	double3 *mpos;
	VEC3D _mp;
	double3 _mf = make_double3(0.0, 0.0, 0.0);
	double3 _mm = make_double3(0.0, 0.0, 0.0);
	if(cy->pointMass())
		_mp = cy->pointMass()->getPosition();
	std::cout << _mp.x << " " << _mp.y << " " << _mp.z << std::endl;
	checkCudaErrors(cudaMalloc((void**)&mpos, sizeof(double3)));
	checkCudaErrors(cudaMemcpy(mpos, &_mp, sizeof(double3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&mforce, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMalloc((void**)&mmoment, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMemset(mforce, 0, sizeof(double3)*ps->numParticle()));
	checkCudaErrors(cudaMemset(mmoment, 0, sizeof(double3)*ps->numParticle()));
	switch (tcm)
	{
	case HMCM: 
		cu_cylinder_hertzian_contact_force(
			0, cy->deviceCylinderInfo(), 
			dpos, dvel, domega, 
			dforce, dmoment, 
			dmass, ps->numParticle(), dcp,
			mpos, mforce, mmoment, _mf, _mm); 
		break;
	case DHS:
		cu_cylinder_hertzian_contact_force(
			1, cy->deviceCylinderInfo(),
			dpos, dvel, domega,
			dforce, dmoment,
			dmass, ps->numParticle(), dcp,
			mpos, mforce, mmoment, _mf, _mm);
		break;
	}
	_mf = reductionD3(mforce, ps->numParticle());
	if (VEC3D(_mf.x, _mf.y, _mf.z).length())
	{
		_mf = _mf;
	}

	if (cy->pointMass()){
		cy->pointMass()->addCollisionForce(VEC3D(_mf.x, _mf.y, _mf.z));
	}
	_mm = reductionD3(mmoment, ps->numParticle());
	if (cy->pointMass()){
		cy->pointMass()->addCollisionMoment(VEC3D(_mm.x, _mm.y, _mm.z));
	}
	checkCudaErrors(cudaFree(mforce)); mforce = NULL;
	checkCudaErrors(cudaFree(mmoment)); mmoment = NULL;
	checkCudaErrors(cudaFree(mpos)); mpos = NULL;
	return true;
}

double collision_particles_cylinder::particle_cylinder_contact_detection(VEC4D& pt, VEC3D& u, VEC3D& cp, unsigned int i)
{
// 	double dist = -1.0f;
// 	VEC3F loc = cy->origin();
// 	VEC3F xp = VEC3F(pt.x, pt.y, pt.z);
// 	VEC3F p1 = cy->basePos();
// 	VEC3F p2 = cy->topPos();
// 	VEC3F ab = p2 - p1;
// 	double t = (xp - p1).dot(ab) / ab.dot();
// 	if (t < 0.0f || t > 1.0f){
// 		cp = p1 + t * ab;
// 		dist = (xp - cp).length();
// 		if (dist < cy->topRadius()){
// 			double overlap = (cp - loc).length();
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
// 		double _r = _at.length();
// 		double r = at.length();		
// 		double th = acos(at.y / r);
// 		double pi = atan(at.x / at.z);
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
// 		double _dist = (cp - xp).length();
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

bool collision_particles_cylinder::collid_with_particle(unsigned int i, double dt)
{
	switch (tcm)
	{
	case HMCM:
		this->HMCModel(i, dt);
		break;
	}
	return true;
}

bool collision_particles_cylinder::DHSModel(unsigned int i, double dt)
{
	VEC4D p = ps->position()[i];
	VEC3D v = ps->velocity()[i];
	VEC3D w = ps->angVelocity()[i];
	VEC3D u, Fn, Ft, M;
	VEC3D sF;
	VEC3D mforce, mmoment;
	VEC3D cp;
	double overlap = particle_cylinder_contact_detection(p, u, cp, i);
	VEC3D si = cp - cy->pointMass()->getPosition();
	//double overlap = (cy->topRadius() + p.w) - dist;
	if (overlap > 0)
	{
		//double rcon = p.w - 0.5f * overlap;
		VEC3D dv = -(v + w.cross(p.w * u));

		constant c = getConstant(p.w, 0.0, ps->mass()[i], 0.0, ps->youngs(), cy->youngs(), ps->poisson(), cy->poisson(), ps->shear(), cy->shear());
		//double collid_dist2 = particle_plane_contact_detection(unit, p, wp, rad);

		double fsn = (-c.kn * pow(overlap, 1.5));
		double fca = cohesionForce(p.w, 0.0, ps->youngs(), 0.0, ps->poisson(), 0.0, fsn);
		double fsd = c.vn * dv.dot(u);
		Fn = (fsn + fca + c.vn * dv.dot(u)) * u;
		//Fn = (fsn + fca + fsd) * unit;// (-c.kn * pow(collid_dist, 1.5f) + c.vn * dv.dot(unit)) * unit;;// fn * unit;
		VEC3D e = dv - dv.dot(u) * u;
		double mag_e = e.length();
		//vector3<double> shf;
		if (mag_e){
			VEC3D s_hat = e / mag_e;
			double ds = mag_e * dt;
			Ft = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * Fn.length()) * s_hat;
			M = (p.w * u).cross(Ft);
		}
		sF = Fn + Ft;
		ps->force()[i] += sF;
		ps->moment()[i] += M;
		mforce = -sF;
		mmoment = -si.cross(sF);
	}
	if (cy->pointMass()){
		cy->pointMass()->addExternalForce(mforce);
		cy->pointMass()->addExternalMoment(mmoment);
	}

	return true;
}

bool collision_particles_cylinder::HMCModel(unsigned int i, double dt)
{
	VEC4D p = ps->position()[i];
	VEC3D v = ps->velocity()[i];
	VEC3D w = ps->angVelocity()[i];
	VEC3D u, Fn, Ft, M;
	VEC3D sF;
	VEC3D mforce, mmoment;
	VEC3D cp;
	double overlap = particle_cylinder_contact_detection(p, u, cp, i);
	VEC3D si = cp - cy->pointMass()->getPosition();
	//double overlap = (cy->topRadius() + p.w) - dist;
	if (overlap > 0)
	{
		VEC3D dv = -(v + w.cross(p.w * u));
		constant c = getConstant(p.w, 0.0, ps->mass()[i], 0.0, ps->youngs(), cy->youngs(), ps->poisson(), cy->poisson(), ps->shear(), cy->shear());
		double fsn = (-c.kn * pow(overlap, 1.5));
		double fca = cohesionForce(p.w, 0.0, ps->youngs(), 0.0, ps->poisson(), 0.0, fsn);
		Fn = (fsn + fca + c.vn * dv.dot(u)) * u;
		VEC3D e = dv - dv.dot(u) * u;
		double mag_e = e.length();
		VEC3D shf;
		if (mag_e){
			VEC3D s_hat = e / mag_e;
			double ds = mag_e * dt;
			Ft = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * Fn.length()) * s_hat;
			M = (p.w * u).cross(Ft);
		}
		sF = Fn + Ft;
		ps->force()[i] += sF;
		ps->moment()[i] += M;
		mforce = -sF;
		mmoment = -si.cross(sF);
	}
	if (cy->pointMass()){
		cy->pointMass()->addExternalForce(mforce);
		cy->pointMass()->addExternalMoment(mmoment);
	}
	
	return true;
}