#ifndef KINEMATICCONSTRAINT_H
#define KINEMATICCONSTRAINT_H

#include <QString>
#include <mphysics_numeric.h>
#include <mphysics_types.h>
#include "modeler.h"

class mass;

class kinematicConstraint
{
public:
	

	kinematicConstraint();
	kinematicConstraint(const kinematicConstraint& _kc);
	kinematicConstraint(modeler *_md, QString&, tKinematicConstraint kt, VEC3D _loc, 
		mass* ip, VEC3D _fi, VEC3D _gi, 
		mass* jp, VEC3D _fj, VEC3D _gj);

	virtual ~kinematicConstraint();

	QString name() const { return nm; }
	mass* iMass() const { return i; }
	mass* jMass() const { return j; }
	VEC3D location() const { return loc; }
	VEC3D axis() const { return ax; }
	tKinematicConstraint constType() const { return type; }
	VEC3D sp_i() const { return spi; }
	VEC3D sp_j() const { return spj; }
	VEC3D f_i() const { return fi; }
	VEC3D f_j() const { return fj; }
	VEC3D h_i() const { return hi; }
	VEC3D h_j() const { return hj; }
	VEC3D g_i() const { return gi; }
	VEC3D g_j() const { return gj; }

	int startRow() const { return srow; }
	int iColumn() const { return icol; }
	int jColumn() const { return jcol; }
	int numConst() const { return nconst; }
	int maxNNZ() const { return maxnnz; }

	void setStartRow(int _sr) { srow = _sr; }
	void setFirstColumn(int _ic) { icol = _ic; }
	void setSecondColumn(int _jc) { jcol = _jc; }
	void setCoordinates();

	//modeler* getModeler() const { return md; }

protected:
	QString nm;
	mass* i;
	mass* j;
	double* lm;
	VEC3D loc;
	VEC3D ax;

	tKinematicConstraint type;
	VEC3D spi, spj;
	VEC3D fi, fj;
	VEC3D hi, hj;
	VEC3D gi, gj;

	int srow;
	int icol;
	int jcol;
	int nconst;
	int maxnnz;

	modeler *md;
};

#endif