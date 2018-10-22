#include "velocity_verlet.h"
#include "model.h"
#include "simulation.h"
#include "vectorTypes.h"
#include "mphysics_cuda_dec.cuh"

velocity_verlet::velocity_verlet()
	: dem_integrator(VELOCITY_VERLET)
{

}

velocity_verlet::~velocity_verlet()
{

}

void velocity_verlet::updatePosition(double *dpos, double* dvel, double* dacc, unsigned int np)
{
	if (simulation::isGpu())
		vv_update_position(dpos, dvel, dacc, np);
	else
	{
		VEC4D_PTR p = (VEC4D_PTR)dpos;
		VEC3D_PTR v = (VEC3D_PTR)dvel;
		VEC3D_PTR a = (VEC3D_PTR)dacc;
		double sqt_dt = 0.5 * simulation::dt * simulation::dt;
		for (unsigned int i = 0; i < np; i++){
			VEC3D old_p = VEC3D(p[i].x, p[i].y, p[i].z);
			VEC3D new_p = old_p + simulation::dt * v[i] + sqt_dt * a[i];
			p[i] = VEC4D(new_p.x, new_p.y, new_p.z, p[i].w);
		}
	}
}

void velocity_verlet::updateVelocity(
	  double *dvel, double* dacc
	, double *domega, double* dalpha
	, double *dforce, double* dmoment
	, double *dmass, double* dinertia, unsigned int np)
{
	if (simulation::isGpu())
		vv_update_velocity(dvel, dacc, domega, dalpha, dforce, dmoment, dmass, dinertia, np);
	else
	{
		double inv_m = 0;
		double inv_i = 0;
		VEC3D_PTR v = (VEC3D_PTR)dvel;
		VEC3D_PTR o = (VEC3D_PTR)domega;
		VEC3D_PTR a = (VEC3D_PTR)dacc;
		VEC3D_PTR aa = (VEC3D_PTR)dalpha;
		VEC3D_PTR f = (VEC3D_PTR)dforce;
		VEC3D_PTR m = (VEC3D_PTR)dmoment;
		for (unsigned int i = 0; i < np; i++){
			inv_m = 1.0 / dmass[i];
			inv_i = 1.0 / dinertia[i];
			v[i] += 0.5 * simulation::dt * a[i];
			o[i] += 0.5 * simulation::dt * aa[i];
			a[i] = inv_m * f[i];
			aa[i] = inv_i * m[i];
			v[i] += 0.5 * simulation::dt * a[i];
			o[i] += 0.5 * simulation::dt * aa[i];
			f[i] = dmass[i] * model::gravity;
			m[i] = 0.0;
		}
	}
}

void velocity_verlet::updatePosition(float *dpos, float* dvel, float* dacc, unsigned int np)
{
	if (simulation::isGpu())
		vv_update_position(dpos, dvel, dacc, np);
}

void velocity_verlet::updateVelocity(
	float *dvel, float* dacc
	, float *domega, float* dalpha
	, float *dforce, float* dmoment
	, float *dmass, float* dinertia, unsigned int np)
{
	if (simulation::isGpu())
		vv_update_velocity(dvel, dacc, domega, dalpha, dforce, dmoment, dmass, dinertia, np);
}