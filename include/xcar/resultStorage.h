#ifndef RESULTSTORAGE_H
#define RESULTSTORAGE_H

#include "colors.h"
#include <QMap>
#include "vectorTypes.h"
#include "types.h"

class resultStorage
{
public:
	typedef struct
	{
		double time;
		VEC3D pos;
		EPD ep;
		VEC3D vel;
		VEC3D omega;
		VEC3D acc;
		VEC3D alpha;
		EPD ea;
		VEC3D tforce;
		VEC3D cforce;
		VEC3D hforce;
	}pointMassResultData;

	typedef struct
	{
		double time;
		VEC3D iAForce;
		VEC4D iRForce;
		VEC3D jAForce;
		VEC4D jRForce;
	}reactionForceData;

	resultStorage();
	~resultStorage();

	static resultStorage* RStorage();

	unsigned int Np();
	unsigned int NFluid();
	unsigned int NFloat();
	unsigned int NBound();
	unsigned int NDummy();
	ucolors::colorMap *ColorMap();
	//void defineUserResult(QString nm);
	double RequriedMemory(unsigned int np, unsigned int npart, solver_type ts);
	void insertTimeData(double ct);
	void insertGLineData(QString tg, VEC3D& b);
	void insertPartName(QString nm);
	void insertPointMassResult(QString& nm, pointMassResultData& pmrd);
	void insertReactionForceResult(QString& nm, reactionForceData& rfd);
	void insertTimeDoubleResult(QString& nm, time_double& td);
	void definePartDatasSPH(bool isOnlyCountNumPart, int idx = -1);
	void definePartDatasDEM(bool isOnlyCountNumPart, int idx = -1);
	void setInformationSPH(unsigned int _np, unsigned int _nf, unsigned int _nft, unsigned int _nb, unsigned int _nd);

	double getPartTime(unsigned int pt);
	double* getPartPosition(unsigned int pt);
	double* getPartVelocity(unsigned int pt);
	double* getPartPressure(unsigned int pt);
	float* getPartPosition_f(unsigned int pt);
	float* getPartVelocity_f(unsigned int pt);
	float* getPartPressure_f(unsigned int pt);
	double* getPartColor(unsigned int pt);
	particle_type* getParticleType();
	bool* getPartFreeSurface(unsigned int pt);
	void allocMemory(unsigned int np, unsigned int npart = 1, solver_type ts = SPH);
	void clearMemory();
	void setResultMemorySPH(unsigned int npart);
	void setResultMemoryDEM(bool _isSingle, unsigned int npart, unsigned int _np);
	void setPartDataFromBinary(unsigned int pt, QString file);
	void openSphResultFiles(QStringList& slist);
	void insertDataSPH(particle_type* tp, double* _p, double* _v, double* _prs, bool isCalcContour = false);
	void exportEachResult2TXT(QString path);
	void openResultList(QString f);

	QMap<QString, QList<VEC3D>>& linePointResults();
	QStringList& partList();
	QMap<QString, QList<pointMassResultData>>& pointMassResults() { return pmrs; }
	QMap<QString, QList<reactionForceData>>& reactionForceResults() { return rfrs; }
	QMap<QString, QList<time_double>>& timeDoubleResults() { return tds; }

private:
	bool isAllocMemory;
	bool isSingle;
	int pdata_type;
	QStringList rList;
	QStringList pList;

	unsigned int total_parts;
	unsigned int cpart;

	unsigned int np;
	unsigned int nfluid;
	unsigned int nfloat;
	unsigned int nbound;
	unsigned int ndummy;

	double maxPress;

	double *time;
	particle_type *type;
	double *pos;
	float *pos_f;
	double *vel;
	float *vel_f;
	double *press;
	float *press_f;
	bool *isf;
	double *color;

	QMap<QString, QList<VEC3D>> glLine;
	QMap<QString, QList<pointMassResultData>> pmrs;
	QMap<QString, QList<reactionForceData>> rfrs;
	QMap<QString, QList<time_double>> tds;

	//QMap<QString, pointMassResultData*> pmrd;

	ucolors::colorMap *cramp;
};

#endif