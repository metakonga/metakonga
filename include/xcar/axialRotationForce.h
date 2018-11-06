#ifndef AXIALROTATIONFORCE_H
#define AXIALROTATIONFORCE_H

#include "forceElement.h"

class axialRotationForce : public forceElement
{
public:
	axialRotationForce();
	axialRotationForce(
		QString _n, mbd_model* _md, VEC3D _loc, VEC3D _u,
		pointMass* _a, pointMass* _b);
	~axialRotationForce();

	void setForceValue(double r_f);

	virtual void calcForce(VECD* rhs);
	virtual void derivate(MATD& lhs, double mul){};
	virtual void derivate_velocity(MATD& lhs, double mul){};
	virtual void saveData(QTextStream& qts);

private:
	double r_force;
	VEC3D loc;
	VEC3D spi;
	VEC3D spj;
	VEC3D u;
};

#endif