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

void velocity_verlet::updatePosition(float dt)
{
	float sqt_dt = 0.5f * dt * dt;
	//float inv_m = 0.f;
	VEC3F _p;
	VEC4F_PTR p = md->particleSystem()->position();
	VEC3F_PTR v = md->particleSystem()->velocity();
	VEC3F_PTR a = md->particleSystem()->acceleration();
	for (unsigned int i = 0; i < md->numParticle(); i++){
		//inv_m = 1.f / md->particleSystem()->mass()[i];
		p[i] += dt * v[i] + sqt_dt * a[i];
		//p[i].plusDataFromVec3(_p);
	}
}

void velocity_verlet::updateVelocity(float dt)
{
	particle_system *ps = md->particleSystem();
	float inv_m = 0; 
	float inv_i = 0;
	VEC3F_PTR v = md->particleSystem()->velocity();
	VEC3F_PTR o = md->particleSystem()->angVelocity();
	VEC3F_PTR a = md->particleSystem()->acceleration();
	VEC3F_PTR aa = md->particleSystem()->angAcceleration();
	VEC3F_PTR f = md->particleSystem()->force();
	VEC3F_PTR m = md->particleSystem()->moment();
	for (unsigned int i = 0; i < md->numParticle(); i++){
		inv_m = 1.f / md->particleSystem()->mass()[i];
		inv_i = 1.f / md->particleSystem()->inertia()[i];
		v[i] += 0.5f * dt * a[i];
		o[i] += 0.5f * dt * aa[i];
		a[i] = inv_m * f[i];
		aa[i] = inv_i * m[i];
		v[i] += 0.5f * dt * a[i];
		o[i] += 0.5f * dt * aa[i];
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