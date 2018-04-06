#ifndef VELOCITY_VERLET_H
#define VELOCITY_VERLET_H

#include "dem_integrator.h"

class velocity_verlet : public dem_integrator
{
public:
	velocity_verlet();
	velocity_verlet(modeler *_md);
	~velocity_verlet();

	//virtual void updatePosition(double dt);
	//virtual void updateVelocity(double dt);
	virtual void updatePosition(double* dpos, double* dvel, double* dacc, unsigned int np);
	virtual void updateVelocity(
		  double *dvel, double* dacc
		, double *domega, double* dalpha
		, double *dforce, double* dmoment
		, double *dmass, double* dinertia, unsigned int np);
};

#endif