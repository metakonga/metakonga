#ifndef AXIALTRANSLATIONFORCE_H
#define AXIALTRANSLATIONFORCE_H

#include "forceElement.h"

class axialTranslationForce : public forceElement
{
public:
	axialTranslationForce();
	axialTranslationForce(
		QString _n, mbd_model* _md, VEC3D _loc, VEC3D _u,
		pointMass* _a, pointMass* _b);
	~axialTranslationForce();

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