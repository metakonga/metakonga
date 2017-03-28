#include "velocity_verlet.h"
#include "modeler.h"
#include "mphysics_cuda_dec.cuh"

velocity_verlet::velocity_verlet()
	: integrator()
{

}

velocity_verlet::velocity_verlet(modeler *_md)
	: integrator(_md)
{

}

velocity_verlet::~velocity_verlet()
{

}

void velocity_verlet::updatePosition(double dt)
{
	double sqt_dt = 0.5 * dt * dt;
	//float inv_m = 0.f;
	VEC3D _p;
	VEC4D_PTR p = md->particleSystem()->position();
	VEC4D_PTR rp = md->particleSystem()->ref_position();
	VEC3D_PTR v = md->particleSystem()->velocity();
	VEC3D_PTR a = md->particleSystem()->acceleration();
	for (unsigned int i = 0; i < md->numParticle(); i++){
		rp[i] = p[i];
		p[i] += dt * v[i] + sqt_dt * a[i];
		//p[i].plusDataFromVec3(_p);
	}
}

void velocity_verlet::updateVelocity(double dt)
{
	particle_system *ps = md->particleSystem();
	double inv_m = 0;
	double inv_i = 0;
	VEC3D_PTR v = md->particleSystem()->velocity();
	VEC3D_PTR o = md->particleSystem()->angVelocity();
	VEC3D_PTR a = md->particleSystem()->acceleration();
	VEC3D_PTR aa = md->particleSystem()->angAcceleration();
	VEC3D_PTR f = md->particleSystem()->force();
	VEC3D_PTR m = md->particleSystem()->moment();
	for (unsigned int i = 0; i < md->numParticle(); i++){
		inv_m = 1.f / md->particleSystem()->mass()[i];
		inv_i = 1.f / md->particleSystem()->inertia()[i];
		v[i] += 0.5 * dt * a[i];
		o[i] += 0.5 * dt * aa[i];
		a[i] = inv_m * f[i];
		aa[i] = inv_i * m[i];
		v[i] += 0.5 * dt * a[i];
		o[i] += 0.5 * dt * aa[i];
	}
}

void velocity_verlet::cuUpdatePosition()
{
	vv_update_position(
		  md->particleSystem()->cuPosition()
		, md->particleSystem()->cuVelocity()
		, md->particleSystem()->cuAcceleration()
		, md->numParticle()
		);
}

void velocity_verlet::cuUpdateVelocity()
{
	vv_update_velocity(
		md->particleSystem()->cuVelocity(),
		md->particleSystem()->cuAcceleration(),
		md->particleSystem()->cuOmega(),
		md->particleSystem()->cuAlpha(),
		md->particleSystem()->cuForce(),
		md->particleSystem()->cuMoment(),
		md->particleSystem()->cuMass(),
		md->particleSystem()->cuInertia(),
		md->numParticle()
		);
}