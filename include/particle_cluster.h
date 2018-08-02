// #ifndef PARTICLE_CLUSTER_H
// #define PARTICLE_CLUSTER_H
// 
// #include "vectorTypes.h"
// #include "stdarg.h"
// 
// class particle_cluster
// {
// public:
// 	particle_cluster();
// 	particle_cluster(unsigned int _nc);
// 	~particle_cluster();
// 
// 	void setIndice(unsigned int sid);
// 	void define(VEC4D* pos, double* _mass, double* _iner);
// 
// 	VEC3D center() { return com; }
// 	static int perCluster() { return nc; }
// 	unsigned int indice(int i) { return c_indice[i]; }
// 	void updatePosition(VEC4D* pp, VEC3D* avp, VEC3D* aap, double dt);
// 	void updateVelocity(VEC3D* v, VEC3D* w, VEC3D* force, VEC3D* moment, double dt);
// 
// 	void setEOM(VEC3D& f, VEC3D& n);
// 	void setTM();
// 	VEC3D angularVelocity_in_globalCoordinate();
// 	VEC3D eulerAngle_dot();
// 	MAT33D op_MTVM(MAT33D& m, MAT33D& v);
// 	static void setConsistNumber(int _nc) { nc = _nc; }
// 
// private:
// 	static int nc;			// the number of particle per cluster
// 	unsigned int *c_indice;		// indice of each particle
// 	VEC3D *local;
// 	VEC3D com;
// 	VEC3D vel;
// 	VEC3D acc;
// 	VEC3D th;
// 	VEC3D dth;
// 	VEC3D ddth;
// 	VEC3D iner;
// 	MAT33D A;
// 	double mass;
// 	double dist;					// distance between com and others
// 	//double rad;
// };
// 
// #endif