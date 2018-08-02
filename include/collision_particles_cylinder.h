// #ifndef COLLISION_PARTICLES_CYLINDER_H
// #define COLLISION_PARTICLES_CYLINDER_H
// 
// #include "collision.h"
// 
// class cylinder;
// class particle_system;
// 
// class collision_particles_cylinder : public collision
// {
// public:
// 	collision_particles_cylinder();
// 	collision_particles_cylinder(QString& _name, modeler* _md, particle_system *_ps, cylinder *_cy, tContactModel _tcm);
// 	virtual ~collision_particles_cylinder();
// 
// 	virtual bool collid(double dt);
// 	virtual bool cuCollid(
// 		double *dpos /* = NULL */, double *dvel /* = NULL  */,
// 		double *domega /* = NULL */, double *dmass /* = NULL  */,
// 		double *dforce /* = NULL  */, double *dmoment /* = NULL */, unsigned int np);
// 	virtual bool collid_with_particle(unsigned int i, double dt);
// 
// private:
// 	double particle_cylinder_contact_detection(VEC4D& xp, VEC3D &u, VEC3D &cp, unsigned int i = 0);
// 	bool HMCModel(unsigned int i, double dt);
// 	bool DHSModel(unsigned int i, double dt);
// 	particle_system *ps;
// 	cylinder *cy;
// };
// 
// #endif