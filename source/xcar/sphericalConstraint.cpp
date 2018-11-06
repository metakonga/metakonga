#include "sphericalConstraint.h"

sphericalConstraint::sphericalConstraint()
	: kinematicConstraint()
{

}

sphericalConstraint::sphericalConstraint(mbd_model* _md, QString& _nm, kinematicConstraint::Type kt,
	pointMass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi,
	pointMass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj)
	: kinematicConstraint(_md, _nm, kt, ip, _spi, _fi, _gi, jp, _spj, _fj, _gj)
{

}

sphericalConstraint::~sphericalConstraint()
{

}

void sphericalConstraint::constraintEquation(double m, double* rhs)
{
	if (ib->NumDOF() == DIM2)
		return;
	else
	{
		VEC3D v3 = jb->Position() + jb->toGlobal(spj) - ib->Position() - ib->toGlobal(spi);
		rhs[srow + 0] = m * v3.x;
		rhs[srow + 1] = m * v3.y;
		rhs[srow + 2] = m * v3.z;
	}
}

void sphericalConstraint::constraintJacobian(SMATD& cjaco)
{
	if (ib->NumDOF() == DIM2)
		return;
	else
	{
		if (ib->MassType() != pointMass::GROUND)
		{
			for (unsigned i(0); i < 3; i++) cjaco(srow + i, icol + i) = -1;
			EPD ep = ib->getEP();// m->getParameterv(ib);
			cjaco.extraction(srow + 0, icol + 3, POINTER(B(ep, -spi)), MAT3X4);
		}
		if (jb->MassType() != pointMass::GROUND)
		{
			//if (!ib->ID()) ib = &ground;
			for (unsigned i(0); i < 3; i++) cjaco(srow + i, jcol + i) = 1;
			EPD ep = jb->getEP();
			cjaco.extraction(srow + 0, jcol + 3, POINTER(B(ep, spj)), MAT3X4);
		}
	}
}

void sphericalConstraint::derivate(MATD& lhs, double mul)
{
	VEC3D L(lm[0], lm[1], lm[2]);
	MAT44D Di = -D(spi, L);
	MAT44D Dj = D(spj, L);

	lhs.plus(icol + 3, icol + 3, POINTER(Di), MAT4x4, mul);
	lhs.plus(jcol + 3, jcol + 3, POINTER(Dj), MAT4x4, mul);
}
