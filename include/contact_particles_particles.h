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

	virtual bool collision(
		double *dpos, double *dvel,
		double *domega, double *dmass,
		double *dforce, double *dmoment,		
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np
		);
	virtual void cudaMemoryAlloc();

private:
	object* obj1;
	object* obj2;
// 	bool hostCollision(
// 		double *dpos /* = NULL */, double *dvel /* = NULL  */,
// 		double *domega /* = NULL */, double *dmass /* = NULL  */,
// 		double *dforce /* = NULL  */, double *dmoment /* = NULL */,
// 		unsigned int np);
	bool hostCollision(
		double *dpos /* = NULL */, double *dvel /* = NULL  */,
		double *domega /* = NULL */, double *dmass /* = NULL  */,
		double *dforce /* = NULL  */, double *dmoment /* = NULL */,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end,
		unsigned int np);
};

#endif