#ifndef VELOCITY_VERLET_H
#define VELOCITY_VERLET_H

#include "mintegrator.h"

class velocity_verlet : public integrator
{
public:
	velocity_verlet();
	velocity_verlet(modeler *_md);
	~velocity_verlet();

	virtual void updatePosition(double dt);
	virtual void updateVelocity(double dt);
	virtual void cuUpdatePosition();
	virtual void cuUpdateVelocity();
};

#endif