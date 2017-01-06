#include "kinematicConstraint.h"
#include "mass.h"

kinematicConstraint::kinematicConstraint()
	: i(NULL)
	, j(NULL)
	, lm(NULL)
	, type(CONSTRAINT)
	//, reactionForce(NULL)
	, srow(0)
	, icol(0)
	, jcol(0)
	, nconst(0)
	, maxnnz(0)
	//, principal_axis(0)
{

}

kinematicConstraint::kinematicConstraint(modeler* _md, QString& _nm, tKinematicConstraint kt, VEC3D _loc, 
										 mass* ip, VEC3D _fi, VEC3D _gi, 
										 mass* jp, VEC3D _fj, VEC3D _gj)
										 : md(_md)
										 , i(ip)
										 , j(jp)
										 , lm(NULL)
										 , type(kt)
										 , nm(_nm)
										 //, reactionForce(NULL)
										 , srow(0)
										 , icol(0)
										 , jcol(0)
										 , nconst(0)
										 , maxnnz(0)
										 , fi(_fi)
										 , fj(_fj)
										 , gi(_gi)
										 , gj(_gj)
										 , loc(_loc)
										 //, principal_axis(0)
{
	hi = fi.cross(gi);
	hj = fj.cross(gj);
	setCoordinates();
}

kinematicConstraint::kinematicConstraint(const kinematicConstraint& _kc)
{
	i = _kc.iMass();
	j = _kc.jMass();
	ax = _kc.axis();
	nm = _kc.name();
	loc = _kc.location();
	type = _kc.constType();
	spi = _kc.sp_i();
	spj = _kc.sp_j();
	hi = _kc.h_i();
	hj = _kc.h_j();
	gi = _kc.g_i();
	gj = _kc.g_j();
	fi = _kc.f_i();
	fj = _kc.f_j();
	srow = _kc.startRow();
	icol = _kc.iColumn();
	jcol = _kc.jColumn();
	nconst = _kc.numConst();
	maxnnz = _kc.maxNNZ();
}

kinematicConstraint::~kinematicConstraint()
{

}

void kinematicConstraint::setCoordinates()
{
	spi = loc - (i ? i->getPosition() : VEC3D(0, 0, 0));
	spj = loc - (j ? j->getPosition() : VEC3D(0, 0, 0));
	spi = i ? i->toLocal(spi) : VEC3D(0, 0, 0);
	spj = j ? j->toLocal(spj) : VEC3D(0, 0, 0);
	switch (type){
	case REVOLUTE:
		nconst = 5;
		(i && j) ? maxnnz += 46 : maxnnz += 23;
		break;
	}
}