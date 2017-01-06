#ifndef BASESIMULATION_H
#define BASESIMULATION_H

#include <QObject>
#include <map>
#include <list>
#include "contactConstant.h"
#include "algebra.h"
#include <QProgressBar>

namespace parview{
	class particles;
	class Object;
}

class BaseSimulation : public QObject
{
	Q_OBJECT

public:
	BaseSimulation()
	{
		gravity.x = 0.0f;
		gravity.y = -9.80665f;
		gravity.z = 0.0f;
	}
	~BaseSimulation(){}

	unsigned int& SaveStep() { return saveStep; }
	float& SimulationTime() { return simTime; }
	float& TimeStep() { return dt; }
	void insertContactObject(parview::Object* obj1, parview::Object* obj2)
	{
		std::map<parview::Object*, parview::Object*>::iterator it = pairContact.find(obj1);
		if (it != pairContact.end()){
			pairContact[obj2] = obj1;
		}
		else{
			pairContact[obj1] = obj2;
		}
	}
	std::map<parview::Object*, parview::Object*>& PairContact() { return pairContact; }
	void ContactConstants(std::list<parview::contactConstant>* cc) { cconsts = cc; }
	QProgressBar* GetProgressBar() { return pBar; }
	QLineEdit* GetDurationTimeWidget() { return durationTime; }

protected:
// 	void setContactCoefficient(cmaterialType m1, cmaterialType m2)
// 	{
// 
// 	}
	virtual void CpuRun() = 0;
	virtual void GpuRun() = 0;
	unsigned int saveStep;
	float simTime;
	float dt;
	algebra::vector3<float> gravity;

	std::map<parview::Object*, parview::Object*> pairContact;
	std::list<parview::contactConstant> *cconsts;

	QProgressBar *pBar;
	QLineEdit *durationTime;
};

#endif