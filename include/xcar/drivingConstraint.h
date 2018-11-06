#ifndef DRIVING_CONSTRAINT_H
#define DRIVING_CONSTRAINT_H

#include <QString>
#include "pointMass.h"
//#include "mphysics_types.h"

class kinematicConstraint;



class drivingConstraint
{
public:
	enum Type{ DRIVING_TRANSLATION = 0, DRIVING_ROTATION };
	drivingConstraint();
	drivingConstraint(QString _name);
	~drivingConstraint();

	void define(kinematicConstraint* kc, Type td, double init, double cont);
	QString& getName() { return name; }
	unsigned int startRow() { return srow; }
//	unsigned int startColumn() { return scol; }
	//bool use(int i) { return use_p[i]; }
	int maxNNZ() { return maxnnz; }
	//pointMass* ActionBody(){ return m; }
	void updateInitialCondition();
	//double constraintEquation(double ct);
	virtual void constraintEquation(double m, double* rhs);
	virtual void constraintJacobian(SMATD& cjaco);
	//virtual void constraintEquation2D(double m) {};

	void setStartRow(unsigned int _sr) { srow = _sr; }
	//void setStartColumn(unsigned int _sc) { scol = _sc; }
	void setStartTime(double st) { start_time = st; }
	void setPlusTime(double pt) { plus_time = pt; }
	void saveData(QTextStream& qts);

// private:
// 	void updateEV(double time);
// 	void updateV(double time);
// 	double vel_rev_CE(double time);
// 	double vel_tra_CE(double time);
// 	void(drivingConstraint::*update_func)(double time);
// 	double(drivingConstraint::*ce_func)(double time);

private:
	double plus_time;
	double start_time;
	int maxnnz;
	double init_v;
	double cons_v;
	double theta;
	unsigned int srow;
	//unsigned int scol;
	QString name;
	kinematicConstraint* kconst;
	Type type;
};

#endif