#ifndef KINEMATICCONSTRAINT_H
#define KINEMATICCONSTRAINT_H

#include <QString>
#include <QTextStream>
#include "algebraMath.h"
#include "pointMass.h"
#include "model.h"
//#include "resultStorage.h"
//#include <mphysics_types.h>
//#include "mbd_model.h"

class mbd_model;
//class pointMass;

class kinematicConstraint
{
public:
	enum Type{ FIXED = 0, SPHERICAL, REVOLUTE, TRANSLATIONAL, UNIVERSAL, CABLE, GEAR, COINCIDE };

	kinematicConstraint();
	kinematicConstraint(QString _nm);
	kinematicConstraint(const kinematicConstraint& _kc);
	kinematicConstraint(mbd_model *_md, QString&, Type kt,
		pointMass* ip, VEC3D& _spi, VEC3D& _fi, VEC3D& _gi, 
		pointMass* jp, VEC3D& _spj, VEC3D& _fj, VEC3D& _gj);

	virtual ~kinematicConstraint();

	QString name() const { return nm; }
	pointMass* iMass() const { return ib; }
	pointMass* jMass() const { return jb; }
	VEC3D axis() const { return ax; }
	Type constType() const { return type; }
	VEC3D sp_i() const { return spi; }
	VEC3D sp_j() const { return spj; }
	VEC3D f_i() const { return fi; }
	VEC3D f_j() const { return fj; }
	VEC3D h_i() const { return hi; }
	VEC3D h_j() const { return hj; }
	VEC3D g_i() const { return gi; }
	VEC3D g_j() const { return gj; }
	VEC3D location() const;
	VEC3D CurrentDistance();

	int startRow() const { return srow; }
	int iColumn() const { return icol; }
	int jColumn() const { return jcol; }
	int numConst() const { return nconst; }
	int maxNNZ() const { return maxnnz; }

	void setLocation(VEC3D& _loc);
	void setStartRow(int _sr) { srow = _sr; }
	void setFirstColumn(int _ic) { icol = _ic; }
	void setSecondColumn(int _jc) { jcol = _jc; }
	void setCoordinates();
//	void calcReactionForceResult(double ct);
	void setLagrangeMultiplierPointer(double* r);

	void saveData(QTextStream& qts);
	void exportResultData2TXT();
	void setZeroLagrangeMultiplier();


	virtual void calculation_reaction_force(double ct) = 0;
	virtual void constraintEquation(double m, double* rhs) = 0;
	virtual void constraintJacobian(SMATD& cjaco) = 0;
	virtual void derivate(MATD& lhs, double mul) = 0;

	//virtual void constraintEquation();

protected:
	QString nm;
	pointMass* ib;
	pointMass* jb;
//	pointMass* k;
	double* lm;
	VEC3D ax;
	VEC3D loc;
	Type type;
	VEC3D spi, spj;
	VEC3D fi, fj;
	VEC3D hi, hj;
	VEC3D gi, gj;

	int srow;
	int icol;
	int jcol;
	int nconst;
	int maxnnz;

	mbd_model *md;

	
};

#endif