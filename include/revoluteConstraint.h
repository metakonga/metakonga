#ifndef REVOLUTECONSTRAINT_H
#define REVOLUTECONSTRAINT_H

#include "kinematicConstraint.h"

class revoluteConstraint : public kinematicConstraint
{
public:
	revoluteConstraint();
	revoluteConstraint(modeler* _md, QString& _nm, tKinematicConstraint kt,
		mass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi,
		mass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj);
	virtual ~revoluteConstraint();


};

#endif