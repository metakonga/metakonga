#include "translationalConstraint.h"

translationalConstraint::translationalConstraint()
	: kinematicConstraint()
{

}

translationalConstraint::translationalConstraint(modeler* _md, QString& _nm, tKinematicConstraint kt, mass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi, mass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj)
	: kinematicConstraint(_md, _nm, kt, ip, _spi, _fi, _gi, jp, _spj, _fj, _gj)
{

}

translationalConstraint::~translationalConstraint()
{

}