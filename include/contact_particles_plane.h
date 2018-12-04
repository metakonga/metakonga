#ifndef CONTACT_PARTICLES_PLANE_H
#define CONTACT_PARTICLES_PLANE_H

#include "contact.h"
#include "plane.h"

class object;
//class contact_particles_cube;

class contact_particles_plane : public contact
{
public:
	contact_particles_plane(const contact* c);
	contact_particles_plane(
		QString _name, contactForce_type t, object* o1, object* o2);
	virtual ~contact_particles_plane();

	void setPlane(plane* _pe);
	virtual void cuda_collision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	virtual void cuda_collision(
		float *pos, float *vel,
		float *omega, float *mass,
		float *force, float *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	virtual void collision(double r, double m, VEC3D& pos, VEC3D& vel, VEC3D& omega, VEC3D& F, VEC3D& M);	                     
	virtual void cudaMemoryAlloc();
	virtual void cudaMemoryAlloc_f();
private:
	double particle_plane_contact_detection(plane* _pe, VEC3D& u, VEC3D& xp, VEC3D& wp, double r);
	friend class contact_particles_cube;
	void singleCollision(
		plane* _pe, double mass, double rad, VEC3D& pos, VEC3D& vel,
		VEC3D& omega, VEC3D& force, VEC3D& moment);
	void cudaMemoryAlloc_planeObject();

	object* p;
	plane *pe;
	device_plane_info *dpi;
	host_plane_info hpi;
	device_plane_info_f *dpi_f;
};

#endif