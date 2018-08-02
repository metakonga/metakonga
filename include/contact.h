#ifndef CONTACT_H
#define CONTACT_H

#include "types.h"
#include "algebraMath.h"
#include "simulation.h"
#include "grid_base.h"

#include "mphysics_cuda_dec.cuh"

class contact
{
public:
	enum pairType{ NO_CONTACT_PAIR = 0, PARTICLE_PARTICLE = 26, PARTICLE_CUBE = 15, PARTICLE_PANE = 16 };
	typedef struct { double kn, vn, ks, vs; }contactParameters;

	contact(const contact* c);
	contact(QString nm, contactForce_type t);
	virtual ~contact();

	QString Name() const { return name; }
	void setContactParameters(double r, double rt, double f);
	double Restitution() const { return restitution; }
	double Friction() const { return friction; }
	double StiffnessRatio() const { return stiffnessRatio; }
	contactForce_type ForceMethod() const { return f_type; }
	material_property_pair MaterialPropertyPair() const { return mpp; }
	device_contact_property* DeviceContactProperty() const { return dcp; }
	pairType PairType() const { return type; }
	
	contactParameters getContactParameters(
		double ir, double jr,
		double im, double jm,
		double iE, double jE,
		double ip, double jp,
		double is, double js);
// 		contactForce_type cft, double rest, double ratio, double fric);
	void setMaterialPair(material_property_pair _mpp) { mpp = _mpp; }
	
	static pairType getContactPair(geometry_type t1, geometry_type t2);

	/*virtual bool collision(double dt) = 0;*/
	virtual bool collision(
		double *dpos, double *dvel, 
		double *domega, double *dmass, 
		double *dforce, double *dmoment, 
		unsigned int *sorted_id, 
		unsigned int *cell_start, 
		unsigned int *cell_end,
		unsigned int np
		) = 0;
	virtual void cudaMemoryAlloc();
	static unsigned int count;

protected:
	void DHSModel
		(contactParameters& c, 
		double cdist, VEC3D& cp, 
		VEC3D& dv, VEC3D& unit, VEC3D& F, VEC3D& M);
	QString name;
	pairType type;
	contactForce_type f_type;
	material_property_pair mpp;
	//contact_parameter cp;
	device_contact_property* dcp;

	double restitution;
	double stiffnessRatio;
	double friction;
};

#endif
// #include <QString>
// #include "mphysics_types.h"
// #include "modeler.h"
// #include "grid_base.h"
// #include "mphysics_cuda_dec.cuh"
// 
// class collision
// {
// public:
// 	collision();
// 	collision(
// 		QString& _name,
// 		modeler* _md, 
// 		QString& o1,
// 		QString& o2, 
// 		tCollisionPair _tp, 
// 		tContactModel _tcm);
// 	collision(const collision& cs);
// 	virtual ~collision();
// 
// 	void allocDeviceMemory();
// 	void setContactParameter(double Ei, double Ej, double Pi, double Pj, double Gi, double Gj, double _rest, double _fric, double _rfric, double _coh, double _ratio);
// 	void setGridBase(grid_base *_gb) { gb = _gb; }
// 	double cohesionForce(double ri, double rj, double Ei, double Ej, double pri, double prj, double Fn);
// 
// 	virtual bool collid(double dt) = 0;
// 	virtual bool cuCollid(
// 		  double *dpos = NULL, double *dvel = NULL
// 		, double *domega = NULL, double *dmass = NULL
// 		, double *dforce = NULL, double *dmoment = NULL, unsigned int np = 0) = 0;
// 	virtual bool collid_with_particle(unsigned int i, double dt) = 0;
// 
// 	constant getConstant(double ir, double jr, double im, double jm, double iE, double jE, double ip, double jp, double si, double sj);
// 	grid_base* getGridBase() { return gb; }
// 	tCollisionPair getCollisionPairType() { return tcp; }
// 
// 	void save_collision_data(QTextStream& ts);
// 
// 	QString& firstObject() { return oname1; }
// 	QString& secondObject() { return oname2; }
// 
// 	double cohesion() { return coh; }
// 
// protected:
// 	QString name;
// 	QString oname1;
// 	QString oname2;
// 	double rest;
// 	double fric;	
// 	double rfric;
// 	double coh;
// 
// 	contact_parameter hcp;				// Host contact parameters
// 	contact_parameter* dcp;				// Device contact parameters
// 
// 	double *td;			// tangential displacement
// 
// 	tContactModel tcm; 
// 	tCollisionPair tcp;
// 
// 	modeler* md;
// 	grid_base* gb;
// };
// 
// #endif