#ifndef COLLISION_PARTICLES_PLANE_H
#define COLLISION_PARTICLES_PLANE_H

#include "collision.h"

class plane;
class particle_system;

class collision_particles_plane : public collision
{
public:
	collision_particles_plane();
	collision_particles_plane(QString& _name, modeler* _md, particle_system *_ps, plane *_p, tContactModel _tcm);
	virtual ~collision_particles_plane();

	virtual bool collid(double dt);
	virtual bool cuCollid();
	virtual bool collid_with_particle(unsigned int i, double dt);

private:
	double particle_plane_contact_detection(VEC3D& u, VEC3D& xp, VEC3D& wp, double r);
	bool HMCModel(unsigned int i, double dt);
	bool DHSModel(unsigned int i, double dt);
	particle_system *ps;
	plane *pe;
};

#endif