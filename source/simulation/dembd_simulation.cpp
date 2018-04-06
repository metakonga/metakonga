#include "dembd_simulation.h"
#include "object.h"
#include "collision.h"
#include <QDebug>
#include <QTime>
#include <QMap>

dembd_simulation::dembd_simulation()
	: simulation()
	, dem(NULL)
	, mbd(NULL)
{

}

dembd_simulation::dembd_simulation(modeler* _md, dem_simulation* _dem, mbd_simulation* _mbd)
	: simulation(_md)
	, dem(_dem)
	, mbd(_mbd)
{

}

dembd_simulation::~dembd_simulation()
{
	if (dem) delete dem; dem = NULL;
	if (mbd) delete mbd; mbd = NULL;
}

bool dembd_simulation::initialize(bool isCpu)
{
	_isWait = false;
	_isWaiting = false;
	_abort = false;
	_interrupt = false;
	nstep = static_cast<unsigned int>((et / dt) + 1);
	mbd->setSimulationCondition(et, dt, step);
	dem->setSimulationCondition(et, dt, step);
   	dem->initialize(false);
 
 	mbd->initialize(isCpu);
	
	dem->cudaDetection();

	collision_dembd(dt);

	return true;
}

void dembd_simulation::collision_dembd(double dt)
{
	for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		m->setCollisionForce(VEC3D(0.0));
		m->setCollisionMoment(VEC3D(0.0));
	}
	//md->particleSystem()->cuParticleCollision();
	foreach(collision* value, md->collisions())
	{
		value->cuCollid();
	}
}

bool dembd_simulation::saveResult(double t, unsigned int part)
{
	dem->cuSaveResult(t, part);
	mbd->saveResult(t, part);
	return true;
}

void dembd_simulation::predictionStep(double dt)
{
	//dem->getIterator()->cuUpdatePosition();
	/*dem->getNeighborhood()->cuDetection();*/
	collision_dembd(dt);
	mbd->prediction(0);
	for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		object* o = it.key();//mass* m = it.value();
		if (o){
			o->updateFromMass();
		}
	}
}

void dembd_simulation::correctionStep(double dt)
{
	unsigned int cnt = 0;
	dem->cudaUpdatePosition();
	dem->cudaDetection();
	while (1){
		collision_dembd(dt);
		double norm = mbd->oneStepCorrection();
		for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
			object* o = it.key();//mass* m = it.value();
			if (o){
				o->updateFromMass();
			}
		}
		if (norm <= 1e-5)
			break;
		if (cnt >= 100){
			std::cout << "WARNING : System equation was not convergence!! (" << norm << ")" << std::endl;
			break;
		}
		cnt++;
		/*std::cout << norm << std::endl;*/
	}
	dem->cudaUpdateVelocity();
}

bool dembd_simulation::cpuRun()
{
	QString msg;
	unsigned int part = 0;
	unsigned int cStep = 0;
	unsigned int eachStep = 0;
	ct = dt * cStep;
	QTextStream qs(&msg);
	qs << qSetFieldWidth(20) << center << "Part" << "S. time" << "P. step" << "T. step" << "E. time";
	cmd->write(CMD_INFO, msg);
	//qs.setFieldWidth(20);
	//qs.setPadChar('-');
	cmd->printLine();
// 	qDebug() << "-------------------------------------------------------------" << endl
// 		<< "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |" << endl
// 		<< "-------------------------------------------------------------";
	if (saveResult(ct, part)){
		
		QTextStream(&msg) << qSetFieldWidth(20) << center << part << ct << eachStep << cStep << 0;
		cmd->write(CMD_INFO, msg);
	}
	QTime tme;
	tme.start();
	QTime startingTime = tme.currentTime();
	QDate startingDate = QDate::currentDate();
	cStep++;
	// nstep = 3;
	dem->cudaDetection();
	while (cStep < nstep)
	{
		if (_isWait)
			//continue;
		if (_abort){
			emit finished();
			return false;
		}
		std::cout << cStep << std::endl;
		ct = dt * cStep;
		md->runExpression(ct, dt);
		predictionStep(dt);
		correctionStep(dt);
		if (!((cStep) % step)){
			part++;
			emit sendProgress(part);
			if (saveResult(ct, part)){
				QTextStream(&msg) << qSetFieldWidth(20) << center << part << ct << eachStep << cStep << tme.elapsed() * 0.001;
				cmd->write(CMD_INFO, msg);
				
			}
			eachStep = 0;

		}
		cStep++;
		eachStep++;
	}
	cmd->printLine();
	QTime endingTime = tme.currentTime();
	QDate endingDate = QDate::currentDate();
	double dtime = tme.elapsed() * 0.001;
	int minute = static_cast<int>(dtime / 60.0);
	int hour = static_cast<int>(minute / 60.0);
	qs.setFieldWidth(0);
	int cgtime = endingTime.second() - startingTime.msec();
	qs  << "     Starting time/date     = " << startingTime.toString() << " / " << startingDate.toString() << endl
		<< "     Ending time/date      = " << endingTime.toString() << " / " << endingDate.toString() << endl
		<< "     CPU + GPU time       = " << dtime << " second  ( " << hour << " h. " << minute << " m. " << dtime << " s. )";

	cmd->write(CMD_INFO, msg);
	cmd->printLine();

	emit finished();
	return true;
}

bool dembd_simulation::gpuRun()
{
	return true;
}