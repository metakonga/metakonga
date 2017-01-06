#include "Euler_integrator.h"
#include "pointmass.h"

using namespace parSIM;

Euler_integrator::Euler_integrator(Simulation *_sim)
	: Integrator(_sim)
{
	
}

Euler_integrator::~Euler_integrator()
{

}

void Euler_integrator::integration()
{
	double inv_mass, inv_iner;
	
	for(unsigned int i = 0; i < np; i++){
		inv_mass = 1 / acc[i].w;
		inv_iner = 1 / alpha[i].w;
		acc[i].x = force[i].x * inv_mass;
		acc[i].y = force[i].y * inv_mass;
		acc[i].z = force[i].z * inv_mass;

		alpha[i].x = moment[i].x * inv_iner;
		alpha[i].y = moment[i].y * inv_iner;
		alpha[i].z = moment[i].z * inv_iner;

		pos[i].x += vel[i].x * dt + 0.5*dt*dt * acc[i].x;
		pos[i].y += vel[i].y * dt + 0.5*dt*dt * acc[i].y;
		pos[i].z += vel[i].z * dt + 0.5*dt*dt * acc[i].z;

		vel[i].x += dt * acc[i].x;
		vel[i].y += dt * acc[i].y;
		vel[i].z += dt * acc[i].z;

		omega[i].x += dt * alpha[i].x;
		omega[i].y += dt * alpha[i].y;
		omega[i].z += dt * alpha[i].z;
	}
	
}

void Euler_integrator::cu_integration()
{
// 	switch(sim->getSolver()){
// 	case MBD:
// 		for(std::map<std::string, pointmass*>::iterator pm = sim->getMasses()->begin(); pm != sim->getMasses()->end(); pm++){
// 			pointmass* m = pm->second;
// 			m->Position() += dt * m->Velocity() + 0.5 * dt * dt * m->Acceleration();
// 			m->Orientation() += dt * m->dOrientation() + 0.5 * dt * dt * m->ddOrientation();
// 			m->Velocity() += dt * m->Acceleration();
// 			m->dOrientation() += dt * m->ddOrientation();
// 			m->MakeTransformationMatrix();
// 			m->cu_update_geometry_data();
// 		}break;
// 	case DEM:
// 		for(unsigned int i = 0; i < np; i++){
// 			inv_mass = 1 / acc[i].w;
// 			inv_iner = 1 / alpha[i].w;
// 			acc[i].x = force[i].x * inv_mass;
// 			acc[i].y = force[i].y * inv_mass;
// 			acc[i].z = force[i].z * inv_mass;
// 
// 			alpha[i].x = moment[i].x * inv_iner;
// 			alpha[i].y = moment[i].y * inv_iner;
// 			alpha[i].z = moment[i].z * inv_iner;
// 
// 			pos[i].x += vel[i].x * dt + 0.5*dt*dt * acc[i].x;
// 			pos[i].y += vel[i].y * dt + 0.5*dt*dt * acc[i].y;
// 			pos[i].z += vel[i].z * dt + 0.5*dt*dt * acc[i].z;
// 
// 			vel[i].x += dt * acc[i].x;
// 			vel[i].y += dt * acc[i].y;
// 			vel[i].z += dt * acc[i].z;
// 
// 			omega[i].x += dt * alpha[i].x;
// 			omega[i].y += dt * alpha[i].y;
// 			omega[i].z += dt * alpha[i].z;
		//}
//	}
}