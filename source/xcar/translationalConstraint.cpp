#include "translationalConstraint.h"
#include <QDebug>

translationalConstraint::translationalConstraint()
	: kinematicConstraint()
{

}

translationalConstraint::translationalConstraint(
	mbd_model* _md, QString& _nm, kinematicConstraint::Type kt,
	pointMass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi, 
	pointMass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj)
	: kinematicConstraint(_md, _nm, kt, ip, _spi, _fi, _gi, jp, _spj, _fj, _gj)
{

}

translationalConstraint::~translationalConstraint()
{

}

void translationalConstraint::calculation_reaction_force(double ct)
{
// 	SMATD jaco;
// 	VECD out;
// 	out.alloc(14);
// 	jaco.alloc(52, 5, 14);
// 	constraintJacobian(jaco);
// 	for (unsigned int i = 0; i < jaco.nnz(); i++)
// 	{
// 		out(jaco.cidx[i]) += jaco.value[i] * lm[jaco.ridx[i] - srow];
// 	}
// 	resultStorage::reactionForceData rfd =
// 	{
// 		ct,
// 		VEC3D(out(0), out(1), out(2)),
// 		VEC4D(out(3), out(4), out(5), out(6)),
// 		VEC3D(out(7), out(8), out(9)),
// 		VEC4D(out(10), out(11), out(12), out(13))
// 	};
// 	model::rs->insertReactionForceResult(nm, rfd);// .push_back(rfd);
}

void translationalConstraint::constraintEquation(double m, double* rhs)
{
	if (ib->NumDOF() == DIM2)
	{

	}
	else
	{
		VEC3D v3 = jb->toGlobal(hj);
		VEC3D v3g = ib->toGlobal(gi);
		VEC3D v3f = ib->toGlobal(fi);
		rhs[srow + 0] = m * v3.dot(v3f);
		rhs[srow + 1] = m * v3.dot(v3g);
		v3 = jb->Position() + jb->toGlobal(spj) - ib->Position();// -ib->toGlobal(kconst->sp_i());
		rhs[srow + 2] = m * (v3.dot(v3f) - spi.dot(fi));
		rhs[srow + 3] = m * (v3.dot(v3g) - spi.dot(gi));
		rhs[srow + 4] = m * v3f.dot(jb->toGlobal(fj));
	}
}

void translationalConstraint::constraintJacobian(SMATD& cjaco)
{
	if (ib->NumDOF() == DIM2)
	{

	}
	else
	{
		VEC3D dij = (jb->Position() + jb->toGlobal(spj)) - (ib->Position() + ib->toGlobal(spi));
		//qDebug() << "constraint : " << nm;
		if (ib->MassType() != pointMass::GROUND)
		{
			//qDebug() << "i body : " << ib->Name();
			
			cjaco.extraction(srow + 0, icol + 3, POINTER(transpose(jb->toGlobal(hj), B(ib->getEP(), fi))), VEC4);
			cjaco.extraction(srow + 1, icol + 3, POINTER(transpose(jb->toGlobal(hj), B(ib->getEP(), gi))), VEC4);
			cjaco.extraction(srow + 2, icol + 0, POINTER((-fi)), POINTER(transpose(dij + ib->toGlobal(spi), B(ib->getEP(), fi))), VEC3_4);
			cjaco.extraction(srow + 3, icol + 0, POINTER((-gi)), POINTER(transpose(dij + ib->toGlobal(spi), B(ib->getEP(), gi))), VEC3_4);
			cjaco.extraction(srow + 4, icol + 3, POINTER(transpose(jb->toGlobal(fj), B(ib->getEP(), fi))), VEC4);
		}
		if (jb->MassType() != pointMass::GROUND)
		{
			//qDebug() << "j body : " << jb->Name();
			cjaco.extraction(srow + 0, jcol + 3, POINTER(transpose(ib->toGlobal(fi), B(jb->getEP(), hj))), VEC4);
			cjaco.extraction(srow + 1, jcol + 3, POINTER(transpose(ib->toGlobal(gi), B(jb->getEP(), hj))), VEC4);
			cjaco.extraction(srow + 2, jcol + 0, POINTER(ib->toGlobal(fi)), POINTER(transpose(ib->toGlobal(fi), B(jb->getEP(), spj))), VEC3_4);
			cjaco.extraction(srow + 3, jcol + 0, POINTER(ib->toGlobal(gi)), POINTER(transpose(ib->toGlobal(gi), B(jb->getEP(), spj))), VEC3_4);
			cjaco.extraction(srow + 4, jcol + 3, POINTER(transpose(ib->toGlobal(fi), B(jb->getEP(), fj))), VEC4);
		}
	}
}

void translationalConstraint::derivate(MATD& lhs, double mul)
{
	MAT44D Dv;
	MAT34D Bv;
// 	bool ig = ib->MassType() != pointMass::GROUND;
// 	bool jg = jb->MassType() != pointMass::GROUND;
	VEC3D dij = (jb->Position() + jb->toGlobal(spj)) - (ib->Position() + ib->toGlobal(spi));

	Dv = lm[0] * D(fi, jb->toGlobal(hj));
	Dv += lm[1] * D(gi, jb->toGlobal(hj));
	Dv += lm[2] * D(fi, dij + ib->toGlobal(spi));
	Dv += lm[3] * D(gi, dij + ib->toGlobal(spi));
	Dv += lm[4] * D(fi, jb->toGlobal(fj));
	lhs.plus(icol + 3, icol + 3, POINTER(Dv), MAT4x4, mul);
	Bv = -lm[2] * B(ib->getEP(), fi) - lm[3] * B(ib->getEP(), gi);
	lhs.plus(icol, icol + 3, POINTER(Bv), MAT3X4, mul);
	lhs.plus(icol + 3, icol, POINTER(Bv), MAT4x3, mul);

	Bv = -Bv;
	lhs.plus(icol + 3, jcol, POINTER(Bv), MAT4x3, mul);
	lhs.plus(jcol, icol + 3, POINTER(Bv), MAT3X4, mul);

	Dv = lm[0] * D(hj, ib->toGlobal(fi));
	Dv += lm[1] * D(hj, ib->toGlobal(gi));
	Dv += lm[2] * D(spj, ib->toGlobal(fi));
	Dv += lm[3] * D(spj, ib->toGlobal(gi));
	Dv += lm[4] * D(fj, ib->toGlobal(fi));
	lhs.plus(jcol + 3, jcol + 3, POINTER(Dv), MAT4x4, mul);

	Dv = lm[0] * transpose(B(ib->getEP(), fi), B(jb->getEP(), hj));
	Dv += lm[1] * transpose(B(ib->getEP(), gi), B(jb->getEP(), hj));
	Dv += lm[2] * transpose(B(ib->getEP(), fi), B(jb->getEP(), spj));
	Dv += lm[3] * transpose(B(ib->getEP(), gi), B(jb->getEP(), spj));
	Dv += lm[4] * transpose(B(ib->getEP(), fi), B(jb->getEP(), fj));
	lhs.plus(icol + 3, jcol + 3, POINTER(Dv), MAT4x4, mul);

	Dv = lm[0] * transpose(B(jb->getEP(), hj), B(ib->getEP(), fi));
	Dv += lm[1] * transpose(B(jb->getEP(), hj), B(ib->getEP(), gi));
	Dv += lm[2] * transpose(B(jb->getEP(), spj), B(ib->getEP(), fi));
	Dv += lm[3] * transpose(B(jb->getEP(), spj), B(ib->getEP(), gi));
	Dv += lm[4] * transpose(B(jb->getEP(), fj), B(ib->getEP(), fi));
	lhs.plus(jcol + 3, icol + 3, POINTER(Dv), MAT4x4, mul);
}
