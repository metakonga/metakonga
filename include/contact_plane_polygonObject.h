#ifndef CONTACT_PLANE_POLYGONOBJECT_H
#define CONTACT_PLANE_POLYGONOBJECT_H

#include "contact.h"
#include "plane.h"
#include "polygonObject.h"

class object;

class contact_plane_polygonObject : public contact
{
public:
	contact_plane_polygonObject(
		QString _name, contactForce_type t, object* o1, object* o2);
	virtual ~contact_plane_polygonObject();

	virtual void cudaMemoryAlloc();
	virtual void collision(double r, double m, VEC3D& pos, VEC3D& vel, VEC3D& omega, VEC3D& F, VEC3D& M);
/*	polygonObject* PolygonObject() { return dynamic_cast<polygonObject*>(po); }*/

private:
	double plane_polygon_contact_detection(plane* _pe, VEC3D& u, VEC3D& xp, VEC3D& wp, double r);
	//friend class contact_particles_cube;
	void singleCollision(
		plane* _pe, double mass, double rad, VEC3D& pos, VEC3D& vel,
		VEC3D& omega, VEC3D& force, VEC3D& moment);
	//object* p;
	plane *pe;
	polygonObject *po;
	device_plane_info *dpi;
	//object* po;
};

#endif