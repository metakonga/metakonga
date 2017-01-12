#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "mphysics_numeric.h"
#include "mphysics_types.h"
#include <QTextStream>
#include <QString>
#include <QList>
#include <QFile>

class object;
class modeler;
class collision;
class grid_base;

class particle_system
{
public:
	particle_system();
	particle_system(QString& _name, modeler *_md);
	~particle_system();

	void clear();
	QString& name() { return nm; }
	VEC4F_PTR position() { return pos; }
	VEC4F_PTR position() const { return pos; }
	VEC3F_PTR velocity() { return vel; }
	VEC3F_PTR velocity() const { return vel; }
	VEC3F_PTR acceleration() { return acc; }
	VEC3F_PTR acceleration() const { return acc; }
	VEC3F_PTR angVelocity() { return omega; }
	VEC3F_PTR angVelocity() const { return omega; }
	VEC3F_PTR angAcceleration() { return alpha; }
	VEC3F_PTR angAcceleration() const { return alpha; }
	VEC3F_PTR force() { return fr; }
	VEC3F_PTR force() const { return fr; }
	VEC3F_PTR moment() { return mm; }
	VEC3F_PTR moment() const { return mm; }
	unsigned int* pairRIV() { return pair_riv; }
	float* relativeImpactVelocity() { return riv; }
	float* mass() { return ms; }
	float* mass() const { return ms; }
	float* inertia() { return iner; }
	float* inertia() const { return iner; }
	//float* radius() { return rad; }
	//float* radius() const { return rad; }
	float maxRadius() { return max_r; }
	float maxRadius() const { return max_r; }

	float* cuPosition() { return d_pos; }
	float* cuVelocity() { return d_vel; }
	float* cuAcceleration() { return d_acc; }
	float* cuOmega() { return d_omega; }
	float* cuAlpha() { return d_alpha; }
	float* cuForce() { return d_fr; }
	float* cuMoment() { return d_mm; }
	//float* cuRadius() { return d_rad; }
	float* cuMass() { return d_ms; }
	float* cuInertia() { return d_iner; }

	float density() { return rho; }
	float density() const { return rho; }
	float youngs() { return E; }
	float youngs() const { return E; }
	float poisson() { return pr; }
	float poisson() const { return pr; }
	float restitution() { return rest; }
	float restitution() const { return rest; }
	float stiffRatio() { return sratio; }
	float stiffRatio() const { return sratio; }
	float friction() { return fric; }
	float friction() const { return fric; }
	float shear() const { return sh; }

	void setPosition(float* vpos);
	void setVelocity(float* vvel);
	bool makeParticles(object *obj, float spacing, float _rad);
	void setCollision(float _rest, float _fric, float _rfric, float _sh);
	void addCollision(collision* c) { cs.push_back(c); }
	void allocMemory(unsigned int _np);
	void resizeMemory(unsigned int _np);
	void cuAllocMemory();
	unsigned int numParticle() { return np; }
	unsigned int numParticle() const { return np; }
	tObject objectType() { return ot; }
	QString baseObject() { return bo; }

	bool particleCollision(float dt);
	void cuParticleCollision(grid_base* gb);
	modeler* getModeler() { return md; }
	modeler* getModeler() const { return md; }
	bool isMemoryAlloc() { return _isMemoryAlloc; }
	void addParticles(object* obj);

	void saveParticleSystem(QFile& oss);
	void setParticlesFromFile(QString& pfile, QString& _bo, unsigned int np, float _rho, float _E, float _pr, float _sh);
	void changeParticlesFromVP(float* _pos);

private:
	bool _isMemoryAlloc;
	QString nm;
	QString bo;					// base object
	unsigned int np;
	float max_r;				// max radius

	//VEC3F_PTR pos = NULL;
	VEC4F_PTR pos = NULL;
	VEC3F_PTR vel = NULL;
	VEC3F_PTR acc = NULL;
	VEC3F_PTR omega = NULL;
	VEC3F_PTR alpha = NULL;
	VEC3F_PTR fr = NULL;
	VEC3F_PTR mm = NULL;
	float* ms = NULL;
	float* iner = NULL;
	unsigned int* pair_riv = NULL;
	float* riv = NULL;			// relative impact velocity
	//float* rad = NULL;

	float* d_pos = NULL;
	float* d_vel = NULL;
	float* d_acc = NULL;
	float* d_omega = NULL;
	float* d_alpha = NULL;
	float* d_fr = NULL;
	float* d_mm = NULL;
	float* d_ms = NULL;
	float* d_iner = NULL;
	float* d_riv = NULL;
	unsigned int* d_pair_riv = NULL;
	//float* d_rad = NULL;

	float rho;
	float E;
	float pr;
	float sh;		//shear modulus

	float rest;
	float sratio;
	float fric;
	float rfric;
	float coh;

	float isc; // initial spacing

	tObject ot;					// object type

	collision *c_p2p;
	QList<collision*> cs;
	modeler *md;
};

#endif