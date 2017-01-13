#ifndef COLLISION_PARTICLES_CYLINDER_H
#define COLLISION_PARTICLES_CYLINDER_H

#include "collision.h"

class cylinder;
class particle_system;

class collision_particles_cylinder : public collision
{
public:
	collision_particles_cylinder();
	collision_particles_cylinder(QString& _name, modeler* _md, particle_system *_ps, cylinder *_cy, tContactModel _tcm);
	virtual ~collision_particles_cylinder();

	virtual bool collid(float dt);
	virtual bool cuCollid();
	virtual bool collid_with_particle(unsigned int i, float dt);

private:
	float particle_cylinder_contact_detection(VEC4F& xp, VEC3F &u, VEC3F &cp, unsigned int i = 0);
	bool HMCModel(unsigned int i, float dt);
	particle_system *ps;
	cylinder *cy;
};

#endif