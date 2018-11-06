#include "fixedConstraint.h"

fixedConstraint::fixedConstraint()
	: kinematicConstraint()
{

}

fixedConstraint::fixedConstraint(mbd_model* _md, QString& _nm, kinematicConstraint::Type kt,
	pointMass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi,
	pointMass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj)
	: kinematicConstraint(_md, _nm, kt, ip, _spi, _fi, _gi, jp, _spj, _fj, _gj)
{

}

fixedConstraint::~fixedConstraint()
{

}

void fixedConstraint::constraintEquation(double m, double* rhs)
{
	if (ib->NumDOF() == DIM2)
	{

	}
	else
	{
		VEC3D v3 = jb->Position() + jb->toGlobal(spj) - ib->Position() - ib->toGlobal(spi);
		rhs[srow + 0] = m * v3.x;
		rhs[srow + 1] = m * v3.y;
		rhs[srow + 2] = m * v3.z;
		v3 = jb->toGlobal(hj);
		rhs[srow + 3] = m * v3.dot(ib->toGlobal(gi));
		rhs[srow + 4] = m * v3.dot(ib->toGlobal(fi));
		rhs[srow + 5] = m * jb->toGlobal(fj).dot(ib->toGlobal(fi));
	}
}

void fixedConstraint::constraintJacobian(SMATD& cjaco)
{
	if (ib->NumDOF() == DIM2)
	{

	}
	else
	{
		if (ib->MassType() != pointMass::GROUND)
		{
			for (unsigned i(0); i < 3; i++) cjaco(srow + i, icol + i) = -1;
			EPD ep = ib->getEP();// m->getParameterv(ib);
			cjaco.extraction(srow + 0, icol + 3, POINTER(B(ep, -spi)), MAT3X4);
			cjaco.extraction(srow + 3, icol + 3, POINTER(transpose(jb->toGlobal(hj), B(ep, gi))), VEC4);
			cjaco.extraction(srow + 4, icol + 3, POINTER(transpose(jb->toGlobal(hj), B(ep, fi))), VEC4);
			cjaco.extraction(srow + 5, icol + 3, POINTER(transpose(jb->toGlobal(fj), B(ep, fi))), VEC4);
		}
		if (jb->MassType() != pointMass::GROUND)
		{
			for (unsigned i(0); i < 3; i++) cjaco(srow + i, jcol + i) = 1;
			EPD ep = jb->getEP();
			cjaco.extraction(srow + 0, jcol + 3, POINTER(B(ep, spj)), MAT3X4);
			cjaco.extraction(srow + 3, jcol + 3, POINTER(transpose(ib->toGlobal(gi), B(ep, hj))), VEC4);
			cjaco.extraction(srow + 4, jcol + 3, POINTER(transpose(ib->toGlobal(fi), B(ep, hj))), VEC4);
			cjaco.extraction(srow + 5, jcol + 3, POINTER(transpose(ib->toGlobal(fi), B(ep, fj))), VEC4);
		}
	}
}

void fixedConstraint::derivate(MATD& lhs, double mul)
{

}
