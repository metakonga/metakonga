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
	
	dem->getNeighborhood()->cuDetection();

	collision_dembd(dt);



	return true;
}

void dembd_simulation::collision_dembd(float dt)
{
	for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		m->setCollisionForce(VEC3D(0.0));
		m->setCollisionMoment(VEC3D(0.0));
	}
	md->particleSystem()->cuParticleCollision(dem->getNeighborhood());
	foreach(collision* value, md->collisions())
	{
		value->cuCollid();
	}
}

bool dembd_simulation::saveResult(float t, unsigned int part)
{
	dem->cuSaveResult(t, part);
	mbd->saveResult(t, part);
	return true;
}

void dembd_simulation::predictionStep(float dt)
{
	dem->getIterator()->cuUpdatePosition();
	dem->getNeighborhood()->cuDetection();
	collision_dembd(dt);
	mbd->prediction(0);
	for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		object* o = it.key();//mass* m = it.value();
		if (o){
			o->updateFromMass();
		}
	}
}

void dembd_simulation::correctionStep(float dt)
{
	collision_dembd(dt);
	unsigned int cnt = 0;
	while (1){
		
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
	dem->getIterator()->cuUpdateVelocity();
}

bool dembd_simulation::cpuRun()
{
	//std::cout << "111" << std::endl;
	unsigned int part = 0;
	unsigned int cStep = 0;
	unsigned int eachStep = 0;

	ct = dt * cStep;
	qDebug() << "-------------------------------------------------------------" << endl
		<< "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |" << endl
		<< "-------------------------------------------------------------";
	QTextStream::AlignRight;
	//QTextStream::setRealNumberPrecision(6);
	QTextStream os(stdout);
	os.setRealNumberPrecision(6);
	if (saveResult(ct, part)){
		os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << "0" << qSetFieldWidth(0) << " |" << endl;
		//std::cout << "| " << std::setw(9) << part << std::setw(12) << ct << std::setw(10) << eachStep << std::setw(11) << cStep << std::setw(15) << 0 << std::endl;
	}
	QTime tme;
	tme.start();
	cStep++;
	// nstep = 3;
	while (cStep < nstep)
	{
		if (_isWait)
			//continue;
		if (_abort){
			emit finished();
			return false;
		}
		//std::cout << cStep << std::endl;
		//std::cout << "111" << std::endl;
// 		if (cStep == 10101)
// 			cStep = 10101;
		ct = dt * cStep;
		md->runExpression(ct, dt);
		predictionStep(dt);
		//std::cout << "111" << std::endl;
		//collision_dembd(dt);
		correctionStep(dt);
		//std::cout << "111" << std::endl;
		if (!((cStep) % step)){
			//mutex.lock();
			part++;
			//pBar->setValue(part);
			emit sendProgress(part);
			if (saveResult(ct, part)){
				os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << tme.elapsed() * 0.001 << qSetFieldWidth(0) << " |" << endl;
			}
			eachStep = 0;

		}
		cStep++;
		eachStep++;
		//std::cout << "111" << std::endl;
	}
	QFile qf(md->modelPath() + "/" + md->modelName() + ".sfi");
	qf.open(QIODevice::WriteOnly);
	QTextStream qt(&qf);
	qt << "moc " << mbd->getOutCount() << endl;
	unsigned int cnt = 0;
	QString objType;
	for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		if (it.value()->ID() == 0)
			continue;
		objType = "object ";
		if (it.key()->objectType() == POLYGON)
			objType = "polygonObject ";
		qt << objType << cnt << " " << it.key()->objectName() << endl;
	}
	qf.close();
	emit finished();
	return true;
}

bool dembd_simulation::gpuRun()
{
	return true;
}