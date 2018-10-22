#ifndef STARTING_MODEL_H
#define STARTING_MODEL_H

#include <QMap>
#include <QFile>
#include "resultStorage.h"

class startingModel
{
public:
	startingModel();
	~startingModel();

	void setEndTime(double _et) { e_time = _et; }
	double endTime() { return e_time; }
	void copyDEMData(unsigned int np, double* p = NULL, double* v = NULL, double* av = NULL);
	void setDEMData(unsigned int np, QFile& qf);
	void setMBDData(QFile& qf);
	double* DEMPosition() { return dem_pos; }
	double* RHS() { return rhs; }
	QMap<QString, resultStorage::pointMassResultData>& MBD_BodyData(){ return mbd_bodies; }

private:
	bool is_alloc_dem;
	bool is_alloc_mbd;
	double e_time;
	double* rhs;
	double* dem_pos;
	double* dem_vel;
	double* dem_avel;

	float* dem_pos_f;
	float* dem_vel_f;
	float* dem_avel_f;

	QMap<QString, resultStorage::pointMassResultData> mbd_bodies;

};

#endif