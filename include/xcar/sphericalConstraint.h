#ifndef SPHERICALCONSTRAINT_H
#define SPHERICALCONSTRAINT_H

#include "kinematicConstraint.h"

class sphericalConstraint : public kinematicConstraint
{
public:
	sphericalConstraint();
	sphericalConstraint(mbd_model* _md, QString& _nm, kinematicConstraint::Type kt,
		pointMass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi,
		pointMass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj);
	virtual ~sphericalConstraint();

	virtual void calculation_reaction_force(double ct) {};
	virtual void constraintEquation(double m, double* rhs);
	virtual void constraintJacobian(SMATD& cjaco);
	virtual void derivate(MATD& lhs, double mul);
};

#endif