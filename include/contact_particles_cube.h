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
	virtual void cudaMemoryAlloc();

private:
// 	bool hostCollision(
// 		double *dpos /* = NULL */, double *dvel /* = NULL  */,
// 		double *domega /* = NULL */, double *dmass /* = NULL  */,
// 		double *dforce /* = NULL  */, double *dmoment /* = NULL */,
// 		unsigned int np);

	object* p;
	cube* cu;
	device_plane_info *dpi;
};

#endif