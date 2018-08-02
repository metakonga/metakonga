#ifndef XDYNAMICSSOLVER_H
#define XDYNAMICSSOLVER_H

#include "simulation.h"
#include "dem_simulation.h"
#include "multibodyDynamics.h"

#include <QThread>

class modelManager;

class xDynamicsSolver : public QThread
{
	Q_OBJECT

public:
	xDynamicsSolver(modelManager* _mg);
	~xDynamicsSolver();

	bool initialize();
	unsigned int totalStep() { return nstep; }
	unsigned int totalPart() { return npart; }

	public slots:
	void setStopCondition();
	
private:
	void run() Q_DECL_OVERRIDE;

	bool savePart(double ct, unsigned int pt);
	
	bool isStop;
	unsigned int nstep;
	unsigned int npart;
	QMutex m_mutex;
	modelManager* mg;
	dem_simulation *dem;
	multibodyDynamics *mbd;

signals:
	void finishedThread();
	void sendProgress(int, QString, QString info = "");
};

#endif