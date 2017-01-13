#include "collision_particles_plane.h"

#include "particle_system.h"
#include "plane.h"
#include "mphysics_cuda_dec.cuh"

collision_particles_plane::collision_particles_plane()
{

}

collision_particles_plane::collision_particles_plane(
	QString& _name, 
	modeler* _md, 
	particle_system *_ps, 
	plane *_p, 
	tContactModel _tcm)
	: collision(_name, _md, _ps->name(), _p->objectName(), PARTICLES_PLANE, _tcm)
	, ps(_ps)
	, pe(_p)
{

}

collision_particles_plane::~collision_particles_plane()
{

}

bool collision_particles_plane::collid(float dt)
{
	return true;
}

bool collision_particles_plane::cuCollid()
{
	switch (tcm)
	{
	case HMCM: cu_plane_hertzian_contact_force(0, pe->devicePlaneInfo(), pe->youngs(), pe->poisson(), pe->shear(), rest, fric, rfric, ps->cuPosition(), ps->cuVelocity(), ps->cuOmega(), ps->cuForce(), ps->cuMoment(), ps->cuMass(), ps->youngs(), ps->poisson(), ps->shear(), ps->numParticle()); break;
	}
	
	return true;
}

float collision_particles_plane::particle_plane_contact_detection(VEC3F& u, VEC3F& xp, VEC3F& wp, float r)
{
	float a_l1 = pow(wp.x - pe->L1(), 2.0f);
	float b_l2 = pow(wp.y - pe->L2(), 2.0f);
	float sqa = wp.x * wp.x;
	float sqb = wp.y * wp.y;
	float sqc = wp.z * wp.z;
	float sqr = r*r;

	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe->L1()) && (wp.y > 0 && wp.y < pe->L2())){
		VEC3F dp = xp - pe->XW();
		vector3<float> uu = pe->UW() / pe->UW().length();
		int pp = -sign(dp.dot(pe->UW()));
		u = pp * uu;
		float collid_dist = r - abs(dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		VEC3F Xsw = xp - pe->XW();
		float h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->L1() && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		VEC3F Xsw = xp - pe->W2();
		float h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->L1() && wp.y > pe->L2() && (a_l1 + b_l2 + sqc) < sqr){
		VEC3F Xsw = xp - pe->W3();
		float h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe->L2() && (sqa + b_l2 + sqc) < sqr){
		VEC3F Xsw = xp - pe->W4();
		float h = Xsw.length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe->L1()) && wp.y < 0 && (sqb + sqc) < sqr){
		VEC3F Xsw = xp - pe->XW();
		VEC3F wj_wi = pe->W2() - pe->XW();
		VEC3F us = wj_wi / wj_wi.length();
		VEC3F h_star = Xsw - (Xsw.dot(us)) * us;
		float h = h_star.length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe->L1()) && wp.y > pe->L2() && (b_l2 + sqc) < sqr){
		VEC3F Xsw = xp - pe->W4();
		VEC3F wj_wi = pe->W3() - pe->W4();
		VEC3F us = wj_wi / wj_wi.length();
		VEC3F h_star = Xsw - (Xsw.dot(us)) * us;
		float h = h_star.length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe->L2()) && wp.x < 0 && (sqr + sqc) < sqr){
		VEC3F Xsw = xp - pe->XW();
		VEC3F wj_wi = pe->W4() - pe->XW();
		VEC3F us = wj_wi / wj_wi.length();
		VEC3F h_star = Xsw - (Xsw.dot(us)) * us;
		float h = h_star.length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe->L2()) && wp.x > pe->L1() && (a_l1 + sqc) < sqr){
		VEC3F Xsw = xp - pe->W2();
		VEC3F wj_wi = pe->W3() - pe->W2();
		VEC3F us = wj_wi / wj_wi.length();
		VEC3F h_star = Xsw - (Xsw.dot(us)) * us;
		float h = h_star.length();
		u = -h_star / h;
		return r - h;
	}

	
	return -1.0f;
}

bool collision_particles_plane::collid_with_particle(unsigned int i, float dt)
{
	switch (tcm)
	{
	case HMCM:
		this->HMCModel(i, dt);
		break;
	}
	return true;
}

bool collision_particles_plane::HMCModel(unsigned int i, float dt)
{
	float ms = ps->mass()[i];
	VEC4F p = ps->position()[i];
	VEC3F v = ps->velocity()[i];
	VEC3F w = ps->angVelocity()[i];
	VEC3F Fn, Ft, M;
	VEC3F dp = p.toVector3() - pe->XW();
	VEC3F wp = VEC3F(dp.dot(pe->U1()), dp.dot(pe->U2()), dp.dot(pe->UW()));
	VEC3F u;
	
	float cdist = particle_plane_contact_detection(u, p.toVector3(), wp, p.w);
	if (cdist > 0){
		float rcon = p.w - 0.5f * cdist;
		VEC3F rv = -(v + w.cross(p.w * u));
		constant c = getConstant(p.w, 0.f, ms, 0.f, ps->youngs(), pe->youngs(), ps->poisson(), pe->poisson(), ps->shear(), pe->shear());
		float fsn = -c.kn * pow(cdist, 1.5f);
		float fdn = c.vn * rv.dot(u);
		Fn = (fsn + fdn) * u;
		VEC3F e = rv - rv.dot(u) * u;
		float mag_e = e.length();
		VEC3F Ft;
		if (mag_e){
			VEC3F sh = e / mag_e;
			float ds = mag_e * dt;
			float fst = -c.ks * ds;
			float fdt = c.vs * rv.dot(sh);
			Ft = (fst + fdt) * sh;
			if (Ft.length() >= c.mu * Fn.length())
				Ft = c.mu * fsn * sh;
			M = (rcon * u).cross(Ft);
			if (w.length()){
				VEC3F on = w / w.length();
				M += -c.rf * fsn * rcon * on;
			}
		}
		ps->force()[i] += Fn;
		ps->moment()[i] += M;


//  		float fsn = (-c.kn * pow(collid_dist, 1.5f));
//  		float fca = cohesionForce(p.w, 0.f, ps->youngs(), 0.f, ps->poisson(), 0.f, fsn);
//  		//float fsd = c.vn * dv.dot(unit);
//  		Fn = (fsn + fca + c.vn * dv.dot(unit)) * unit;
// 		//Fn = (fsn + fca + fsd) * unit;// (-c.kn * pow(collid_dist, 1.5f) + c.vn * dv.dot(unit)) * unit;;// fn * unit;
// 		vector3<float> e = dv - dv.dot(unit) * unit;
// 		float mag_e = e.length();
// 		vector3<float> shf;
// 		if (mag_e){
// 			vector3<float> s_hat = e / mag_e;
// 			float ds = mag_e * dt;
// 			Ft = min(c.ks * ds + c.vs * dv.dot(s_hat), c.mu * Fn.length()) * s_hat;
// 			M = (p.w * unit).cross(Ft);
// 		}
// 		//mf = sf + shf;
// 		ps->force()[i] += Fn + Ft;
// 		ps->moment()[i] += M;
	}

	//ps->force()[i] += mf;
	//ps->moment()[i] += mm;
	return true;
}