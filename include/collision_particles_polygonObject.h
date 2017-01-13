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

	virtual bool collid(float dt);
	virtual bool cuCollid();
	virtual bool collid_with_particle(unsigned int i, float dt);

	

private:
	VEC3F particle_polygon_contact_detection(host_polygon_info& hpi, VEC3F& p, float pr);
	particle_system *ps;
	polygonObject *po;
};

#endif