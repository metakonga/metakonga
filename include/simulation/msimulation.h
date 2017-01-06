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

class simulation : public QObject
{
	Q_OBJECT

public:
	simulation();
	simulation(modeler* _md);
	virtual ~simulation();

	modeler* model() { return md; }
	virtual bool initialize(bool isCpu) = 0;
	
	

//	QProgressBar* GetProgressBar() { return pBar; }
	//QLineEdit* GetDurationTimeWidget() { return durationTime; }
	
	void setSimulationCondition(float _et, float _dt, unsigned int _step) { et = _et; dt = _dt; step = _step; }
	void setWaitSimulation(bool _isw) { _isWait = _isw; }
	bool isWaiting() { return _isWaiting; }
	bool wait() { return _isWait; }
	bool interrupt() { return _interrupt; }
	void abort();
	unsigned int numStep() { return nstep; }
	void reverseWait();
	
	
protected:
	modeler* md;
	grid_base* gb;
	integrator* itor;

	QMutex mutex;
	QWaitCondition condition;
	
	//QLineEdit *durationTime;

	static float ct;
	float et, dt;
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