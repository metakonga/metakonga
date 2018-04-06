#ifndef COLLISION_H
#define COLLISION_H

#include <QString>
#include "mphysics_types.h"
#include "modeler.h"
#include "grid_base.h"
#include "mphysics_cuda_dec.cuh"

class collision
{
public:
	collision();
	collision(
		QString& _name,
		modeler* _md, 
		QString& o1,
		QString& o2, 
		tCollisionPair _tp, 
		tContactModel _tcm);
	collision(const collision& cs);
	virtual ~collision();

	void allocDeviceMemory();
	void setContactParameter(double Ei, double Ej, double Pi, double Pj, double Gi, double Gj, double _rest, double _fric, double _rfric, double _coh, double _ratio);
	void setGridBase(grid_base *_gb) { gb = _gb; }
	double cohesionForce(double ri, double rj, double Ei, double Ej, double pri, double prj, double Fn);

	virtual bool collid(double dt) = 0;
	virtual bool cuCollid(
		  double *dpos = NULL, double *dvel = NULL
		, double *domega = NULL, double *dmass = NULL
		, double *dforce = NULL, double *dmoment = NULL, unsigned int np = 0) = 0;
	virtual bool collid_with_particle(unsigned int i, double dt) = 0;

	constant getConstant(double ir, double jr, double im, double jm, double iE, double jE, double ip, double jp, double si, double sj);
	grid_base* getGridBase() { return gb; }
	tCollisionPair getCollisionPairType() { return tcp; }

	void save_collision_data(QTextStream& ts);

	QString& firstObject() { return oname1; }
	QString& secondObject() { return oname2; }

	double cohesion() { return coh; }

protected:
	QString name;
	QString oname1;
	QString oname2;
	double rest;
	double fric;	
	double rfric;
	double coh;

	contact_parameter hcp;				// Host contact parameters
	contact_parameter* dcp;				// Device contact parameters

	double *td;			// tangential displacement

	tContactModel tcm; 
	tCollisionPair tcp;

	modeler* md;
	grid_base* gb;
};

#endif