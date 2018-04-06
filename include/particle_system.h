#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#include "mphysics_numeric.h"
#include "mphysics_types.h"
#include "particle_cluster.h"
#include <QTextStream>
#include <QString>
#include <QList>
#include <QFile>
#include <QVector>

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
	VEC4D_PTR position() { return pos; }
	//VEC4D_PTR initPosition() { return pos0; }
	VEC4D_PTR position() const { return pos; }
	VEC3D_PTR velocity() { return vel; }
	VEC3D_PTR velocity() const { return vel; }
	VEC3D_PTR acceleration() { return acc; }
	VEC3D_PTR acceleration() const { return acc; }
	VEC3D_PTR angVelocity() { return omega; }
	VEC3D_PTR angVelocity() const { return omega; }
	VEC3D_PTR angAcceleration() { return alpha; }
	VEC3D_PTR angAcceleration() const { return alpha; }
	VEC3D_PTR force() { return fr; }
	VEC3D_PTR force() const { return fr; }
	VEC3D_PTR moment() { return mm; }
	VEC3D_PTR moment() const { return mm; }
	unsigned int* pairRIV() { return pair_riv; }
	double* relativeImpactVelocity() { return riv; }
	double* mass() { return ms; }
	double* mass() const { return ms; }
	double* inertia() { return iner; }
	double* inertia() const { return iner; }
	//double* radius() { return rad; }
	//double* radius() const { return rad; }
	double maxRadius() { return max_r; }
	double maxRadius() const { return max_r; }

// 	double* cuPosition() { return d_pos; }
// 	double* cuVelocity() { return d_vel; }
// 	double* cuAcceleration() { return d_acc; }
// 	double* cuOmega() { return d_omega; }
// 	double* cuAlpha() { return d_alpha; }
// 	double* cuForce() { return d_fr; }
// 	double* cuMoment() { return d_mm; }
// 	//double* cuRadius() { return d_rad; }
// 	double* cuMass() { return d_ms; }
// 	double* cuInertia() { return d_iner; }

	double density() { return rho; }
	double density() const { return rho; }
	double youngs() { return E; }
	double youngs() const { return E; }
	double poisson() { return pr; }
	double poisson() const { return pr; }
	///double restitution() { return rest; }
	///double restitution() const { return rest; }
	//double stiffRatio() { return sratio; }
	//double stiffRatio() const { return sratio; }
	////double friction() { return fric; }
	//double friction() const { return fric; }
	double shear() const { return sh; }
	//double cohesion() const { return coh; }

	void setPosition(double* vpos);
	void setVelocity(double* vvel);
	//void setCohesion(double _coh) { coh = _coh; }
	//void setStiffnessRatio(double _ratio) { sratio = _ratio; }
	bool makeParticles(object *obj, VEC3UI size, VEC3D spacing, double _rad, unsigned int nstack);
	//void setCollision(double _rest, double _fric, double _rfric, double _sh, double _ratio);
	//void addCollision(collision* c) { cs.push_back(c); }
	void allocMemory(unsigned int _np);
	void resizeMemory(unsigned int _np);
	void resizeMemoryForStack(unsigned int _np);
	//void resizeCudaMemory(unsigned int _np);
	bool updateStackParticle(double ct, tSolveDevice tDev = CPU);
	void cuAllocMemory();
	unsigned int numParticle() { return np; }
	unsigned int numParticle() const { return np; }
	//unsigned int numStackParticle() { return total_stack_particle; }
	unsigned int numParticlePerStack() { return npPerStack; }
	tObject objectType() { return ot; }
	QString baseObject() { return bo; }
	tGenerationParticleMethod generationMethod() { return tGenParticle; }

	//bool particleCollision(double dt);
	//void cuParticleCollision();
	modeler* getModeler() { return md; }
	modeler* getModeler() const { return md; }
	bool isMemoryAlloc() { return _isMemoryAlloc; }
	void addParticles(object* obj, VEC3UI size);

	void saveParticleSystem(QFile& oss);
	void setParticlesFromFile(QString& pfile, QString& _bo, unsigned int np, double _rho, double _E, double _pr, double _sh);
	void setGenerationMethod(tGenerationParticleMethod t_gpm, unsigned int _nStack, double _stack_dt, unsigned int _pstack) { tGenParticle = t_gpm; nStack = _nStack; stack_dt = _stack_dt; npPerStack = _pstack; }
	void changeParticlesFromVP(double* _pos);
	//particle_cluster* particleClusters(unsigned int id) { return &(pc[id]); }
	void clusterUpdatePosition(double dt);
	void clusterUpdateVelocity(double dt);
	particle_cluster* getParticleClusterFromParticleID(unsigned int id) { return pc.at(id); }
	void setParticleCluster(int _consist);
	QVector<particle_cluster*>& particleCluster() { return pc; }
	void appendCluster();

private:
	bool _isMemoryAlloc;
	QString nm;
	QString bo;					// base object
	unsigned int np;
	double max_r;				// max radius

	//VEC3D_PTR pos = NULL;
	VEC4D_PTR pos = NULL;
//	VEC4D_PTR pos0 = NULL;
	VEC3D_PTR vel = NULL;
	VEC3D_PTR acc = NULL;
	VEC3D_PTR omega = NULL;
	VEC3D_PTR alpha = NULL;
	VEC3D_PTR fr = NULL;
	VEC3D_PTR mm = NULL;
	unsigned int *cid = NULL;
	double* ms = NULL;
	double* iner = NULL;
	unsigned int* pair_riv = NULL;
	double* riv = NULL;			// relative impact velocity
	//double* rad = NULL;

// 	double* d_pos = NULL;
// 	double* d_vel = NULL;
// 	double* d_acc = NULL;
// 	double* d_omega = NULL;
// 	double* d_alpha = NULL;
// 	double* d_fr = NULL;
// 	double* d_mm = NULL;
// 	double* d_ms = NULL;
// 	double* d_iner = NULL;
// 	double* d_riv = NULL;
// 	unsigned int* d_pair_riv = NULL;
	//double* d_rad = NULL;

	QVector<particle_cluster*> pc;				// particle clusters

	double rho;
	double E;
	double pr;
	double sh;		//shear modulus

	VEC3D isc; // initial spacing
	VEC3UI genParticleSize;

	tObject ot;					// object type
	tGenerationParticleMethod tGenParticle;
	unsigned int nStack;
	unsigned int cStack;
	unsigned int npPerStack;
	double last_stack_time;
	double stack_dt;

	modeler *md;
};

#endif