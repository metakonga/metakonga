#ifndef MBD_SIMULATION_H
#define MBD_SIMULATION_H

#include "msimulation.h"
#include "mass.h"
#include <mkl.h>

QT_BEGIN_NAMESPACE
class QFile;
QT_END_NAMESPACE

class mbd_simulation : public simulation
{
	typedef QMap<QString, kinematicConstraint*>::iterator kConstIterator;
	typedef QMap<QString, drivingConstraint*>::iterator dConstIterator;
	typedef QMap<object*, mass*>::iterator massIterator;
public:
	mbd_simulation();
	mbd_simulation(modeler *_md);
	virtual ~mbd_simulation();

	virtual bool initialize(bool isCpu);

	void prediction(unsigned int cs);
	double correction(unsigned int cs);
	double oneStepCorrection();

	bool saveResult(double ct, unsigned int p);

	unsigned int getOutCount() { return outCount; }

private:
	unsigned int mdim;
	unsigned int tdim;
	unsigned int nnz; 
	int dof;				// degree of freedom

	MKL_INT lapack_one;
	MKL_INT lapack_info;
	MKL_INT *ptDof;
	MKL_INT *permutation;

	double dt2accp;
	double dt2accv;
	double dt2acc;
	double divalpha;
	double divbeta;
	double alpha;
	double beta;
	double gamma;
	double eps;
	double *lagMul;

	VECD rhs;
	MATD lhs;
	VECD pre;
	VECD ipp;
	VECD ipv;
	VECD ee;
	VECD cEQ;
	SMATD cjaco;
	mass ground;

	void calculateRhs();
	void FULL_LEOM();
	void calcMassMatrix(double mul = 1.0);
	void calcForceVector(VECD *vec = NULL);
	VEC4D calcInertiaForce(EPD& ev, MAT33D& J, EPD& ep);
	VEC4D calcMoment(EPD& ep, VEC3D& m);
	void sparseConstraintJacobian();
	
	void calcMassSystemJacobian(double mul);
	void calcConstraintSystemJacobian(double mul);
	
	MAT44D D(VEC3D& a, VEC3D& lag);
	void constraintEquation();

	QFile qf_out;
	unsigned int outCount;

public slots:
	virtual bool cpuRun();
	virtual bool gpuRun();
};

#endif