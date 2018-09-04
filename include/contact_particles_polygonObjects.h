#ifndef CONTACT_PARTICLES_POLYGONOBJECTS_H
#define CONTACT_PARTICLES_POLYGONOBJECTS_H

#include "contact.h"
#include <QMap>

class polygonObject;
class contact_particles_polygonObject;

class contact_particles_polygonObjects : public contact
{
	enum polygonContactType { FACE = 0, VERTEX, EDGE };
public:
	contact_particles_polygonObjects();
	~contact_particles_polygonObjects();

	double MaxRadiusOfPolySphere() { return maxRadius; }
	VEC4D* HostSphereData() { return hsphere; }
	unsigned int define(QMap<QString, contact_particles_polygonObject*>& cppos);
	bool cppolyCollision(
		unsigned int idx, double r, double m,
		VEC3D& p, VEC3D& v, VEC3D& o, VEC3D& F, VEC3D& M);
	unsigned int NumContact() { return ncontact; }
	void setNumContact(unsigned int c) { ncontact = c; }

	virtual void cudaMemoryAlloc();
	
private:
	VEC3D particle_polygon_contact_detection(host_polygon_info& dpi, VEC3D& p, double r, polygonContactType& _pct);
// 	bool hostCollision(
// 		double *dpos, double *dvel,
// 		double *domega, double *dmass,
// 		double *dforce, double *dmoment,
// 		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end,
// 		unsigned int np);

	unsigned int ncontact;
	polygonContactType *pct;
	double maxRadius;
	unsigned int nPobjs;
	unsigned int npolySphere;
 	contact_parameter* hcp;
// 	contact_parameter* dcp;
	VEC4D *hsphere;
	double* dsphere;
	host_polygon_info* hpi;
	device_polygon_info* dpi;

	QMap<unsigned int, polygonObject*> pair_ip;
};

#endif