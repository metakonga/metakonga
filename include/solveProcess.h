#ifndef SOLVEPROCESS_H
#define SOLVEPROCESS_H

#include <QObject>
#include <QWaitCondition>
#include <QMutex>

class dem_simulation;
class modeler;

class solveProcess : public QObject
{
	Q_OBJECT

public:
	explicit solveProcess(/*modeler *_md, double _et, double _dt, unsigned int _step, */QObject *parent = 0);
	void abort();

private:
	bool isAbort;
	bool isInterrupt;
	QMutex mutex;
	QWaitCondition condition;

	dem_simulation *dem;
	modeler* md;
	double et;
	double dt;
	double step;

signals:
	void valueChanged(const QString &value);
	void finished();
	void spSignal();

public slots:
	void mainLoop();
};

#endif