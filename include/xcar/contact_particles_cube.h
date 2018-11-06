#ifndef CONTACT_PARTICLES_CUBE_H
#define CONTACT_PARTICLES_CUBE_H

#include "contact.h"

class cube;
class object;

class contact_particles_cube : public contact
{
public:
	contact_particles_cube(QString _name, contactForce_type t, object* o1, object* o2);
	virtual ~contact_particles_cube();

	virtual void collision(
		double r, double m, VEC3D& pos, VEC3D& vel, VEC3D& omega, VEC3D& F, VEC3D& M);
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
	virtual void cudaMemoryAlloc();
	virtual void cudaMemoryAlloc_f();
private:
// 	bool hostCollision(
// 		double *dpos /* = NULL */, double *dvel /* = NULL  */,
// 		double *domega /* = NULL */, double *dmass /* = NULL  */,
// 		double *dforce /* = NULL  */, double *dmoment /* = NULL */,
// 		unsigned int np);

	object* p;
	cube* cu;
	device_plane_info *dpi;
	device_plane_info_f *dpi_f;
};

#endif