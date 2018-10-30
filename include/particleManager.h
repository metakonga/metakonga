#ifndef PARTICLEMANAGER_H
#define PARTICLEMANAGER_H

#include "algebraMath.h"
#include "types.h"
//#include "particle_cluster.h"
#include "object.h"
#include <QTextStream>
#include <QString>
#include <QList>
#include <QFile>
#include <QVector>

class particleManager
{
public:
	typedef struct
	{
		unsigned int sid;
		unsigned int np;
		double min_radius, max_radius;
		double youngs, density, poisson, shear;
		VEC3D loc;
		VEC3D dim;
		VEC3D dir;
	}particlesInfo;
	particleManager();
	//particleManager(QString& _name);
	~particleManager();

	void Save(QTextStream& qts);
	void Open(QTextStream& qts);

	QMap<QString, QString>& Logs() { return logs; }
	QMap<QString, particlesInfo>& ParticleInfomations() { return pinfos; }
	unsigned int Np() { return np; }
	unsigned int RealTimeCreating() { return is_realtime_creating; }
	void setRealTimeCreating(bool b) { is_realtime_creating = b; }
	bool ChangedParticleModel() { return is_changed_particle; }
	bool OneByOneCreating() { return one_by_one; }
	unsigned int NumCreatingPerSecond() { return per_np; }
	double TimeCreatingPerGroup() { return per_time; }
	unsigned int NextCreatingPerGroup();
	object* Object() { return obj; }
	double* Position() { return (double*)pos; }
	float* Position_f() { return (float*)pos_f; }
	QString setParticleDataFromPart(QString& f);

	VEC4D* CreateCubeParticle(
		QString n, material_type type, unsigned int nx, unsigned int ny, unsigned int nz,
		double lx, double ly, double lz,
		double spacing, double min_radius, double max_radius,
		double youngs, double density, double poisson, double shear
		);

	VEC4D* CreatePlaneParticle(
		QString n, material_type type, unsigned int nx, unsigned int ny, unsigned int _np,
		double lx, double ly, double lz,
		double dx, double dy, double dz,
		double spacing, double min_radius, double max_radius,
		double youngs, double density, double poisson, double shear, bool isr = false, unsigned int pnp = 0, double pnt = 0.0, bool obo = false);
	VEC4D* CreateCircleParticle(
		QString n, material_type type, double cdia, unsigned int _np, 
		double lx, double ly, double lz, 
		double dx, double dy, double dz,
		double spacing, double min_radius, double max_radius,
		double youngs, double density, double poisson, double shear, bool isr = false, unsigned int pnp = 0, double pnt = 0.0, bool obo = false);
	static unsigned int count;

private:
	double calcMass(double r);

private:
	bool is_realtime_creating;
	bool is_changed_particle;
	bool one_by_one;
	unsigned int per_np;
	unsigned int np;
	double per_time;
	object *obj;
	VEC4D *pos;
	VEC4F *pos_f;

	QMap<QString, particlesInfo> pinfos;
	QList<unsigned int> np_group;
	QList<unsigned int>::iterator np_group_iterator;
	QMap<QString, QString> logs;
};

#endif