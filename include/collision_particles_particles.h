#ifndef COLLISION_PARTICLES_PARTICLES_H
#define COLLISION_PARTICLES_PARTICLES_H

#include "particle_system.h"
#include "collision.h"

class particle_system;

class collision_particles_particles : public collision
{
public:
	collision_particles_particles();
	collision_particles_particles(QString& _name, modeler* _md, particle_system* _ps, tContactModel _tcm);
	virtual ~collision_particles_particles();

	virtual bool collid(double dt);
	virtual bool cuCollid(
		double *dpos /* = NULL */, double *dvel /* = NULL  */,
		double *domega /* = NULL */, double *dmass /* = NULL  */,
		double *dforce /* = NULL  */, double *dmoment /* = NULL */, unsigned int np);
	virtual bool collid_with_particle(unsigned int i, double dt) { return true; }

	bool HMCModel(double dt);
	bool DHSModel(double dt);

private:
	particle_system *ps;
};

#endif