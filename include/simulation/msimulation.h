#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "modeler.h"
#include "grid_base.h"
#include "mintegrator.h"
#include <QObject>
#include <QMutex>
#include <QLineEdit>
#include <QProgressBar>
#include <QWaitCondition>
#include "cmdWindow.h"


class simulation : public QObject
{
	Q_OBJECT

public:
	simulation();
	simulation(modeler* _md);
	virtual ~simulation();

	modeler* model() { return md; }
	virtual bool initialize(bool isCpu) = 0;
	
	void setSimulationCondition(double _et, double _dt, unsigned int _step) { et = _et; dt = _dt; step = _step; }
	void setWaitSimulation(bool _isw) { _isWait = _isw; }
	bool isWaiting() { return _isWaiting; }
	bool wait() { return _isWait; }
	bool interrupt() { return _interrupt; }
	void abort();
	unsigned int numStep() { return nstep; }
	void reverseWait();
	void setCommandWindow(cmdWindow* _cmd) { cmd = _cmd; }
	
protected:
	modeler* md;
	grid_base* gb;
	integrator* itor;
	cmdWindow* cmd;

	QMutex mutex;
	QWaitCondition condition;
	
	//QLineEdit *durationTime;

	static double ct;
	double et, dt;
	unsigned int step;
	unsigned int nstep;

	bool _isWaiting;
	bool _isWait;
	bool _abort;
	bool _interrupt;

public slots:
	virtual bool cpuRun() = 0;
	virtual bool gpuRun() = 0;
	

signals:
	void finished();
	void sendProgress(unsigned int);
};

#endif