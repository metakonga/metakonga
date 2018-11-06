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
	float MaxRadiusOfPolySphere_f() { return maxRadius_f; }
	double* SphereData() { return dsphere; }
	VEC4D* HostSphereData() { return hsphere; }
	float* SphereData_f() { return dsphere_f; }
	unsigned int define(QMap<QString, contact_particles_polygonObject*>& cppos);
	unsigned int define_f(QMap<QString, contact_particles_polygonObject*>& cppos);
	bool cppolyCollision(
		unsigned int idx, double r, double m,
		VEC3D& p, VEC3D& v, VEC3D& o, VEC3D& F, VEC3D& M);
	unsigned int NumContact() { return ncontact; }
	void setNumContact(unsigned int c) { ncontact = c; }
	void updatePolygonObjectData();
	void updatePolygonObjectData_f();
	virtual void cudaMemoryAlloc();
	virtual void cudaMemoryAlloc_f();
	virtual void cuda_collision(
		double *pos, double *vel, double *omega, 
		double *mass, double *force, double *moment, 
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);
	virtual void cuda_collision(
		float *pos, float *vel, float *omega,
		float *mass, float *force, float *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);
	void setZeroCollisionForce();
	
private:
	VEC3D particle_polygon_contact_detection(host_polygon_info& dpi, VEC3D& p, double r, polygonContactType& _pct);

	unsigned int ncontact;
	polygonContactType *pct;
	double maxRadius;
	float maxRadius_f;
	unsigned int nPobjs;
	unsigned int npolySphere;
 	contact_parameter* hcp;
// 	contact_parameter* dcp;
	VEC4D *hsphere;
	VEC4F *hsphere_f;
	double* dsphere;
	float* dsphere_f;
	host_polygon_info* hpi;
	host_polygon_info_f* hpi_f;
	device_polygon_info* dpi;
	device_polygon_info_f* dpi_f;
	QMap<unsigned int, polygonObject*> pair_ip;
};

#endif