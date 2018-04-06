#include "msimulation.h"
#include <QDebug>
#include <QThread>
#include "object.h"

double simulation::ct = 0.0;

simulation::simulation()
	: md(NULL)
	, et(0)
	, dt(0)
	, step(0)
	, _isWait(false)
	, _abort(false)
	, _interrupt(false)
	, nstep(0)
{

}

simulation::simulation(modeler* _md)
	: md(_md)
	, et(0)
	, dt(0)
	, step(0)
	, _isCpu(false)
	, _isWait(false)
	, _abort(false)
	, _interrupt(false)
	, nstep(0)
{
	md->saveModeler();
//	md->closeStream();
}

simulation::~simulation()
{

}

void simulation::abort()
{
//	qDebug() << "Request worker aborting in Thread " << thread()->currentThreadId();
// 	QMutexLocker locker(&mutex);
 	_abort = true;
// 	condition.wakeOne();
}

void simulation::reverseWait()
{
	_isWait = _isWait ? false : true;
}

