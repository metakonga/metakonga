#include "revoluteConstraint.h"

revoluteConstraint::revoluteConstraint()
	: kinematicConstraint()
{

}

revoluteConstraint::revoluteConstraint(modeler* _md, QString& _nm, tKinematicConstraint kt, VEC3D _loc, mass* ip, VEC3D _fi, VEC3D _gi, mass* jp, VEC3D _fj, VEC3D _gj)
	: kinematicConstraint(_md, _nm, kt, _loc, ip, _fi, _gi, jp, _fj, _gj)
{

}

revoluteConstraint::~revoluteConstraint()
{

}