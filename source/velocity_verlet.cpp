#include "velocity_verlet.h"
#include "modeler.h"
#include "mphysics_cuda_dec.cuh"

velocity_verlet::velocity_verlet()
	: dem_integrator()
{

}

velocity_verlet::velocity_verlet(modeler *_md)
	: dem_integrator(_md)
{

}

velocity_verlet::~velocity_verlet()
{

}

// void velocity_verlet::updatePosition(double dt)
// {
// 	double sqt_dt = 0.5 * dt * dt;
// 	//float inv_m = 0.f;
// 	VEC3D _p;
// 	VEC4D_PTR p = md->particleSystem()->position();
// 	//VEC4D_PTR rp = md->particleSystem()->ref_position();
// 	VEC3D_PTR v = md->particleSystem()->velocity();
// 	VEC3D_PTR a = md->particleSystem()->acceleration();
// 	for (unsigned int i = 0; i < md->numParticle(); i++){
// 		//rp[i] = p[i];
// 		p[i] += dt * v[i] + sqt_dt * a[i];
// 		//p[i].plusDataFromVec3(_p);
// 	}
// }

// void velocity_verlet::updateVelocity(double dt)
// {
// 	particle_system *ps = md->particleSystem();
// 	double inv_m = 0;
// 	double inv_i = 0;
// 	VEC3D_PTR v = md->particleSystem()->velocity();
// 	VEC3D_PTR o = md->particleSystem()->angVelocity();
// 	VEC3D_PTR a = md->particleSystem()->acceleration();
// 	VEC3D_PTR aa = md->particleSystem()->angAcceleration();
// 	VEC3D_PTR f = md->particleSystem()->force();
// 	VEC3D_PTR m = md->particleSystem()->moment();
// 	for (unsigned int i = 0; i < md->numParticle(); i++){
// 		inv_m = 1.f / md->particleSystem()->mass()[i];
// 		inv_i = 1.f / md->particleSystem()->inertia()[i];
// 		v[i] += 0.5 * dt * a[i];
// 		o[i] += 0.5 * dt * aa[i];
// 		a[i] = inv_m * f[i];
// 		aa[i] = inv_i * m[i];
// 		v[i] += 0.5 * dt * a[i];
// 		o[i] += 0.5 * dt * aa[i];
// 	}
// }

void velocity_verlet::updatePosition(double *dpos, double* dvel, double* dacc, unsigned int np)
{
	if (md->solveDevice() == GPU)
		vv_update_position(dpos, dvel, dacc, np);
	else
	{
		double sqt_dt = 0.5 * dt * dt;
		for (unsigned int i = 0; i < np; i++){
			dpos[i] += dt * dvel[i] + sqt_dt * dacc[i];
		}
	}

}

void velocity_verlet::updateVelocity(
	  double *dvel, double* dacc
	, double *domega, double* dalpha
	, double *dforce, double* dmoment
	, double *dmass, double* dinertia, unsigned int np)
{
	if (md->solveDevice() == GPU)
		vv_update_velocity(dvel, dacc, domega, dalpha, dforce, dmoment, dmass, dinertia, np);
	else
	{
		particle_system *ps = md->particleSystem();
		double inv_m = 0;
		double inv_i = 0;
		VEC3D_PTR v = (VEC3D_PTR)dvel;
		VEC3D_PTR o = (VEC3D_PTR)domega;
		VEC3D_PTR a = (VEC3D_PTR)dacc;
		VEC3D_PTR aa = (VEC3D_PTR)dalpha;
		VEC3D_PTR f = (VEC3D_PTR)dforce;
		VEC3D_PTR m = (VEC3D_PTR)dmoment;
		for (unsigned int i = 0; i < md->numParticle(); i++){
			inv_m = 1.f / dmass[i];
			inv_i = 1.f / dinertia[i];
			v[i] += 0.5 * dt * a[i];
			o[i] += 0.5 * dt * aa[i];
			a[i] = inv_m * f[i];
			aa[i] = inv_i * m[i];
			v[i] += 0.5 * dt * a[i];
			o[i] += 0.5 * dt * aa[i];
		}
	}
}