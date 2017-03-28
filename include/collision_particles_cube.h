#ifndef COLLISION_PARTICLES_CUBE_H
#define COLLISION_PARTICLES_CUBE_H

#include "collision.h"

class cube;
class particle_system;

class collision_particles_cube : public collision
{
public:
	collision_particles_cube();
	collision_particles_cube(QString& _name, modeler* _md, particle_system *_ps, cube *_c, tContactModel _tcm);
	virtual ~collision_particles_cube();

	virtual bool collid(double dt);
	virtual bool cuCollid();
	virtual bool collid_with_particle(unsigned int i, double dt);

private:
	bool HMCModel(unsigned inti, double dt);
	particle_system *ps;
	cube *cu;
};

#endif