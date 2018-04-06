#ifndef COLLISION_PARTICLES_POLYGONOBJECT_H
#define COLLISION_PARTICLES_POLYGONOBJECT_H

#include "collision.h"

class polygonObject;
class particle_system;

class collision_particles_polygonObject : public collision
{
public:
	collision_particles_polygonObject();
	collision_particles_polygonObject(QString& _name, modeler* _md, particle_system *_ps, polygonObject * _poly, tContactModel _tcm);
	virtual ~collision_particles_polygonObject();

	virtual bool collid(double dt);
	virtual bool cuCollid(
		double *dpos /* = NULL */, double *dvel /* = NULL  */,
		double *domega /* = NULL */, double *dmass /* = NULL  */,
		double *dforce /* = NULL  */, double *dmoment /* = NULL */, unsigned int np);
	virtual bool collid_with_particle(unsigned int i, double dt);	

private:
	VEC3D particle_polygon_contact_detection(host_polygon_info& hpi, VEC3D& p, double pr);
	particle_system *ps;
	polygonObject *po;
};

#endif