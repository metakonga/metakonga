#ifndef TRANSLATIONALCONSTRAINT_H
#define TRANSLATIONALCONSTRAINT_H

#include "kinematicConstraint.h"

class translationalConstraint : public kinematicConstraint
{
public:
	translationalConstraint();
	translationalConstraint(modeler* _md, QString& _nm, tKinematicConstraint kt,
		mass* ip, VEC3D& spi, VEC3D& _fi, VEC3D& _gi,
		mass* jp, VEC3D& spj, VEC3D& _fj, VEC3D& _gj);
	virtual ~translationalConstraint();


};

#endif