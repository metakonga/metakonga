#ifndef COLLISION_PARTICLES_PARTICLES_H
#define COLLISION_PARTICLES_PARTICLES_H

#include "particle_system.h"
#include "collision.h"

class particle_system;

class collision_particles_particles : public collision
{
public:
	collision_particles_particles();
	collision_particles_particles(QString& _name, modeler* _md, particle_system* _ps);
	virtual ~collision_particles_particles();

	virtual bool collid(float dt);
	virtual bool cuCollid(){ return true; }
	virtual bool collid_with_particle(unsigned int i, float dt) { return true; }

	bool HMCModel(float dt);

private:
	particle_system *ps;
};

#endif