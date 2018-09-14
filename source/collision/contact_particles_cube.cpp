#include "contact_particles_cube.h"
#include "contact_particles_plane.h"
#include "cube.h"
#include "object.h"

contact_particles_cube::contact_particles_cube
(QString _name, contactForce_type t, object* o1, object* o2)
	: contact(_name, t)
	, cu(NULL)
{
	cu = dynamic_cast<cube*>((o1->ObjectType() == CUBE ? o1 : o2));
	p = o1->ObjectType() != CUBE ? o1 : o2;
}

contact_particles_cube::~contact_particles_cube()
{

}

void contact_particles_cube::collision(
	double r, double m, VEC3D& pos, VEC3D& vel, VEC3D& omega, VEC3D& F, VEC3D& M)
{
	contact_particles_plane cpps(this);
	plane* planes = cu->Planes();
	for (unsigned int j = 0; j < 6; j++)
	{
		VEC3D m_f, m_m;
		cpps.singleCollision(planes + j, m, r, pos, vel, omega, m_f, m_m);
		F += m_f;
		M += m_m;
	}
}

void contact_particles_cube::cuda_collision(
	double *pos, double *vel, double *omega, 
	double *mass, double *force, double *moment, 
	unsigned int *sorted_id, unsigned int *cell_start, 
	unsigned int *cell_end, unsigned int np)
{
	cu_cube_contact_force(1, dpi, pos, vel, omega, force, moment, mass, np, dcp);
}

void contact_particles_cube::cudaMemoryAlloc()
{
	contact::cudaMemoryAlloc();
	device_plane_info *_dpi = new device_plane_info[6];
	plane* pl = cu->Planes();
	for (unsigned i = 0; i < 6; i++)
	{
		plane pe = pl[i];
		_dpi[i].l1 = pe.L1();
		_dpi[i].l2 = pe.L2();
		_dpi[i].xw = make_double3(pe.XW().x, pe.XW().y, pe.XW().z);
		_dpi[i].uw = make_double3(pe.UW().x, pe.UW().y, pe.UW().z);
		_dpi[i].u1 = make_double3(pe.U1().x, pe.U1().y, pe.U1().z);
		_dpi[i].u2 = make_double3(pe.U2().x, pe.U2().y, pe.U2().z);
		_dpi[i].pa = make_double3(pe.PA().x, pe.PA().y, pe.PA().z);
		_dpi[i].pb = make_double3(pe.PB().x, pe.PB().y, pe.PB().z);
		_dpi[i].w2 = make_double3(pe.W2().x, pe.W2().y, pe.W2().z);
		_dpi[i].w3 = make_double3(pe.W3().x, pe.W3().y, pe.W3().z);
		_dpi[i].w4 = make_double3(pe.W4().x, pe.W4().y, pe.W4().z);
	}

	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(device_plane_info) * 6));
	checkCudaErrors(cudaMemcpy(dpi, _dpi, sizeof(device_plane_info) * 6, cudaMemcpyHostToDevice));
	delete [] _dpi;
}
// 
// bool contact_particles_cube::hostCollision(
// 	double *m_pos, double *m_vel, double *m_omega,
// 	double *m_mass, double *m_force, double *m_moment,
// 	unsigned int np)
// {
// 	VEC4D* pos = (VEC4D*)m_pos;
// 	VEC3D* vel = (VEC3D*)m_vel;
// 	VEC3D* omega = (VEC3D*)m_omega;
// 	VEC3D* force = (VEC3D*)m_force;
// 	VEC3D* moment = (VEC3D*)m_moment;
// 	double* mass = m_mass;
// 
// 	contact_particles_plane cpps(this);
// 	plane* planes = cu->Planes();
// 	for (unsigned int i = 0; i < np; i++)
// 	{
// 		double ms = mass[i];
// 		double rad = pos[i].w;
// 		VEC3D p = VEC3D(pos[i].x, pos[i].y, pos[i].z);
// 		VEC3D v = vel[i];
// 		VEC3D w = omega[i];
// 		VEC3D F, M;
// 		for (unsigned int j = 0; j < 6; j++)
// 		{
// 			cpps.singleCollision(planes + j, ms, rad, p, v, w, F, M);
// 			force[i] += F;
// 			moment[i] += M;
// 		}
// 	}
// 	return true;
// }

// 
// collision_particles_cube::~collision_particles_cube()
// {
// 
// }
// 
// bool collision_particles_cube::collid(double dt)
// {
// 	return true;
// }
// 
// bool collision_particles_cube::cuCollid(
// 	double *dpos, double *dvel, 
// 	double *domega, double *dmass, 
// 	double *dforce, double *dmoment, unsigned int np)
// {
// 	return true;
// }
// 
// bool collision_particles_cube::collid_with_particle(unsigned int i, double dt)
// {
// 	switch (tcm)
// 	{
// 	case HMCM:
// 		this->HMCModel(i, dt);
// 		break;
// 	}
// 	return true;
// }
// 
// bool collision_particles_cube::HMCModel(unsigned int i, double dt)
// {
// // 	VEC3F unit;
// // 	double rad = ps->radius()[i];
// // 	double ms = ps->mass()[i];
// // 	VEC3F p = ps->position()[i];
// // 	VEC3F v = ps->velocity()[i];
// // 	VEC3F w = ps->angVelocity()[i];
// // 	cohesion = 0.f;
// // 	VEC3F Fn, Ft, M;
// // 	for (int i = 0; i < 6; i++)
// // 	VEC3F dp = p - pe->XW();
// // 	VEC3F wp = VEC3F(dp.dot(pe->U1()), dp.dot(pe->U2()), dp.dot(pe->UW()));
// 	
// 	return true;
// }