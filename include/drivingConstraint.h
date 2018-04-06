#ifndef DRIVING_CONSTRAINT_H
#define DRIVING_CONSTRAINT_H

#include <QString>
#include "mass.h"
#include "mphysics_types.h"

class kinematicConstraint;

class drivingConstraint
{
public:
	drivingConstraint();
	drivingConstraint(QString& _name);
	~drivingConstraint();

	void define(kinematicConstraint* kc, tDriving td, double val);
	void driving(double time);
	QString& getName() { return name; }
	size_t startRow() { return srow; }
	size_t startColumn() { return scol; }
	bool use(int i) { return use_p[i]; }
	int maxNNZ() { return maxnnz; }

	double constraintEquation(double ct);

	void setStartRow(size_t _sr) { srow = _sr; }

private:
	void updateEV(double time);
	void updateV(double time);
	double vel_rev_CE(double time);
	double vel_tra_CE(double time);
	void(drivingConstraint::*update_func)(double time);
	double(drivingConstraint::*ce_func)(double time);

private:
	bool use_p[7];
	int maxnnz;
	size_t srow;
	size_t scol;
	QString name;
	VEC3D direction;
	VEC3D mple_t;
	VEC3D init_t;
	EPD mple_e;
	EPD init_e;
	kinematicConstraint* kconst;
	tDriving dtype;
	mass* m;
};

#endif