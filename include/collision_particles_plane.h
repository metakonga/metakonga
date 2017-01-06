#ifndef COLLISION_PARTICLES_PLANE_H
#define COLLISION_PARTICLES_PLANE_H

#include "collision.h"

class plane;
class particle_system;

class collision_particles_plane : public collision
{
public:
	collision_particles_plane();
	collision_particles_plane(QString& _name, modeler* _md, particle_system *_ps, plane *_p);
	virtual ~collision_particles_plane();

	virtual bool collid(float dt);
	virtual bool cuCollid();
	virtual bool collid_with_particle(unsigned int i, float dt);

private:
	float particle_plane_contact_detection(VEC3F& u, VEC3F& xp, VEC3F& wp, float r);
	bool HMCModel(unsigned int i, float dt);
	particle_system *ps;
	plane *pe;
};

#endif