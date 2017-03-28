#ifndef INTEGRATOR_H
#define INTEGRATOR_H

class modeler;

class integrator
{
public:
	integrator();
	integrator(modeler* _md);
	virtual ~integrator();

	virtual void updatePosition(double dt) = 0;
	virtual void updateVelocity(double dt) = 0;
	virtual void cuUpdatePosition() = 0;
	virtual void cuUpdateVelocity() = 0;

protected:
	modeler *md;
};

#endif