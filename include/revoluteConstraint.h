#ifndef REVOLUTECONSTRAINT_H
#define REVOLUTECONSTRAINT_H

#include "kinematicConstraint.h"

class revoluteConstraint : public kinematicConstraint
{
public:
	revoluteConstraint();
	revoluteConstraint(modeler* _md, QString& _nm, tKinematicConstraint kt, VEC3D _loc,
		mass* ip, VEC3D _fi, VEC3D _gi,
		mass* jp, VEC3D _fj, VEC3D _gj);
	virtual ~revoluteConstraint();


};

#endif