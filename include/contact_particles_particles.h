#ifndef CONTACT_PARTICLES_PARTICLES_H
#define CONTACT_PARTICLES_PARTICLES_H

#include "contact.h"

class object;
 
class contact_particles_particles : public contact
{
public:
	contact_particles_particles(
		QString _name, contactForce_type t, object* o1, object* o2);
	virtual ~contact_particles_particles();

	void cppCollision(
		double ir, double jr, 
		double im, double jm,
		VEC3D& ip, VEC3D& jp, 
		VEC3D& iv, VEC3D& jv, 
		VEC3D& io, VEC3D& jo, 
		VEC3D& F, VEC3D& M);
	virtual void cudaMemoryAlloc();

private:
	object* obj1;
	object* obj2;
// 	bool hostCollision(
// 		double *dpos /* = NULL */, double *dvel /* = NULL  */,
// 		double *domega /* = NULL */, double *dmass /* = NULL  */,
// 		double *dforce /* = NULL  */, double *dmoment /* = NULL */,
// 		unsigned int np);
// 	bool hostCollision(
// 		double *dpos /* = NULL */, double *dvel /* = NULL  */,
// 		double *domega /* = NULL */, double *dmass /* = NULL  */,
// 		double *dforce /* = NULL  */, double *dmoment /* = NULL */,
// 		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end,
// 		unsigned int np);
};

#endif