#include "Verlet_integrator.h"
#include "cu_dem_dec.cuh"
#include "particle.h"

using namespace parSIM;

Verlet_integrator::Verlet_integrator(Simulation *_sim)
	: Integrator(_sim)
	, seq(false)
{

}

Verlet_integrator::~Verlet_integrator()
{

}

 void Verlet_integrator::integration()
{

	if(!seq){
		for(unsigned int i = 0; i < np; i++){
			pos[i].x += vel[i].x * dt + 0.5*dt*dt * acc[i].x;
			pos[i].y += vel[i].y * dt + 0.5*dt*dt * acc[i].y;
			pos[i].z += vel[i].z * dt + 0.5*dt*dt * acc[i].z;
		}
		seq = true;
	}else
	{
		vector3<double> m_vel;
		vector3<double> m_acc;
		vector3<double> m_omega;
		vector3<double> m_alpha;

		for(unsigned int i = 0; i < np; i++){
			m_vel = vector3<double>(vel[i].x, vel[i].y, vel[i].z) + 0.5 * dt * vector3<double>(acc[i].x, acc[i].y, acc[i].z);
			m_omega = vector3<double>(omega[i].x, omega[i].y, omega[i].z) + 0.5 * dt * vector3<double>(alpha[i].x, alpha[i].y, alpha[i].z);
			m_acc = force[i] / acc[i].w;
			m_alpha = moment[i] / alpha[i].w;
			m_vel += 0.5 * dt * m_acc;
			m_omega += 0.5 * dt * m_alpha;
			vel[i] = vector4<double>(m_vel.x, m_vel.y, m_vel.z, vel[i].w);
			omega[i] = vector4<double>(m_omega.x, m_omega.y, m_omega.z, omega[i].w);
			acc[i] = vector4<double>(m_acc.x, m_acc.y, m_acc.z, acc[i].w);
			alpha[i] = vector4<double>(m_alpha.x, m_alpha.y, m_alpha.z, alpha[i].w);
		}
		
		seq = false;
	}
}

 void Verlet_integrator::cu_integration()
 {
	 if(!seq){
		 vv_update_position(d_pos, d_vel, d_acc, np);
		 seq = true;
	 }
	 else{
		 vv_update_velocity(d_vel, d_acc, d_omega, d_alpha, d_force, d_moment, np);
		 seq = false;
	 }
 }