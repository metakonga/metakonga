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

	virtual bool collision(
		double *dpos, double *dvel,
		double *domega, double *dmass,
		double *dforce, double *dmoment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);
	
	virtual void cudaMemoryAlloc();

private:
	bool hostCollision(
		double *dpos /* = NULL */, double *dvel /* = NULL  */,
		double *domega /* = NULL */, double *dmass /* = NULL  */,
		double *dforce /* = NULL  */, double *dmoment /* = NULL */,
		unsigned int np);

	object* p;
	cube* cu;
	device_plane_info *dpi;
};

#endif