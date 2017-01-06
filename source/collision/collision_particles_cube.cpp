#include "collision_particles_cube.h"
#include "particle_system.h"
#include "cube.h"

collision_particles_cube::collision_particles_cube()
{

}

collision_particles_cube::collision_particles_cube(QString& _name, modeler* _md, particle_system *_ps, cube *_c)
	: collision(_name, _md, _ps->name(), _c->objectName(), PARTICLES_CUBE)
	, ps(_ps)
	, cu(_c)
{

}

collision_particles_cube::~collision_particles_cube()
{

}

bool collision_particles_cube::collid(float dt)
{
	return true;
}

bool collision_particles_cube::cuCollid()
{
	return true;
}

bool collision_particles_cube::collid_with_particle(unsigned int i, float dt)
{
	switch (tcm)
	{
	case HMCM:
		this->HMCModel(i, dt);
		break;
	}
	return true;
}

bool collision_particles_cube::HMCModel(unsigned int i, float dt)
{
// 	VEC3F unit;
// 	float rad = ps->radius()[i];
// 	float ms = ps->mass()[i];
// 	VEC3F p = ps->position()[i];
// 	VEC3F v = ps->velocity()[i];
// 	VEC3F w = ps->angVelocity()[i];
// 	cohesion = 0.f;
// 	VEC3F Fn, Ft, M;
// 	for (int i = 0; i < 6; i++)
// 	VEC3F dp = p - pe->XW();
// 	VEC3F wp = VEC3F(dp.dot(pe->U1()), dp.dot(pe->U2()), dp.dot(pe->UW()));
	
	return true;
}