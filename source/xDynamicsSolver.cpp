#include "xDynamicsSolver.h"
#include "modelManager.h"
#include "errors.h"
#include <QTime>
#include <QDebug>

xDynamicsSolver::xDynamicsSolver(modelManager* _mg)
	: mg(_mg)
	, dem(NULL)
	, mbd(NULL)
	, isStop(false)
{
	if (mg->DEMModel())
	{
		dem = new dem_simulation(mg->DEMModel());
	}
	if (mg->MBDModel())
	{
		mbd = new multibodyDynamics(mg->MBDModel());
	}
}

xDynamicsSolver::~xDynamicsSolver()
{
	if (dem) delete dem; dem = NULL;
	if (mbd) delete mbd; mbd = NULL;
}

bool xDynamicsSolver::initialize()
{
	
	nstep = static_cast<unsigned int>((simulation::et / simulation::dt) + 1);
	npart = static_cast<unsigned int>((nstep / simulation::st) + 1);
	if (dem)
	{
		unsigned int np = mg->DEMModel()->ParticleManager()->Np();
		dem->initialize(mg->ContactManager());
		model::rs->setResultMemoryDEM(npart, np);
	}
	if (mbd)
	{
		mbd->initialize();
	}
	
	savePart(0, 0);
	return true;
}

void xDynamicsSolver::setStopCondition()
{
	m_mutex.lock();
	isStop = true;
	m_mutex.unlock();
}

bool xDynamicsSolver::savePart(double ct, unsigned int pt)
{
	if (dem)
	{
		model::rs->insertTimeData(ct);
		double *v_pos = model::rs->getPartPosition(pt);
		double *v_vel = model::rs->getPartVelocity(pt);
		QString part_name = dem->saveResult(v_pos, v_vel, ct, pt);
		model::rs->insertPartName(part_name);
		model::rs->definePartDatasDEM(false, pt);
	}
	if (mbd)
	{
		mbd->saveResult(ct);
	}
	return true;
}

void xDynamicsSolver::run()
{
	isStop = false;
	unsigned int part = 0;
	unsigned int cstep = 0;
	unsigned int eachStep = 0;
	unsigned int numPart = 0;
	double dt = simulation::dt;
	double ct = dt * cstep;
	bool isUpdated = false;
	simulation::setCurrentTime(ct);
	double total_time = 0.0;
	QString ch;
	QTextStream qts(&ch);
	qts.setFieldAlignment(QTextStream::FieldAlignment::AlignRight);
	qts << "|" 
		<< "        Num. Part   "
		<< " S. Time      " 
		<< "  I. Part        "  
		<< "  I. Total      " 
		<< "E. Time          "
		<< "|";
		//<< "I. NR |";
//	cmd->printLine();
	sendProgress(0, "__line__");
	sendProgress(part, ch); 
	ch.clear();
	
// 	qts << "| " 
// 		<< qSetFieldWidth(14) << part 
// 		<< qSetFieldWidth(17) << ct 
// 		<< qSetFieldWidth(12) << eachStep 
// 		<< qSetFieldWidth(13) << cstep 
// 		<< qSetFieldWidth(22) << 0 
// 		<< "|";
	QTime tme;
	//savePart(ct, part);
	tme.start();
	QTime startingTime = tme.currentTime();
	QDate startingDate = QDate::currentDate();
	bool mbd_state = true;
	while (cstep < nstep)
	{
		QMutexLocker locker(&m_mutex);
		if (isStop)
			break;
		cstep++;
		eachStep++;
		ct += simulation::dt;
#ifdef _DEBUG
		qDebug() << ct;
#endif
		simulation::setCurrentTime(ct);
		if (dem) 
			dem->oneStepAnalysis();
		if (mbd)
		{
			mbd_state = mbd->oneStepAnalysis(ct, cstep);
			if (!mbd_state)
			{
				errors::Error(mbd->MbdModel()->modelName());
				break;
			}
		}
			
		if (!((cstep) % simulation::st))
		{
			double dur_time = tme.elapsed() * 0.001;
			total_time += dur_time;
			part++;
			if (savePart(ct, part))
			{
				ch.clear(); 
				qts << qSetFieldWidth(0) << "| "
					<< qSetFieldWidth(nFit(15, part)) << part
					<< qSetFieldWidth(nFit(20, ct)) << ct
					<< qSetFieldWidth(nFit(20, eachStep)) << eachStep
					<< qSetFieldWidth(nFit(20, cstep)) << cstep
					<< qSetFieldWidth(nFit(20, dur_time)) << dur_time
// 					<< qSetFieldWidth(17) << ct 
// 					<< qSetFieldWidth(12) << eachStep 
// 					<< qSetFieldWidth(13) << cstep 
// 					/*<< qSetFieldWidth(22)*/ << dur_time 
					<< "|";
				sendProgress(part, ch); 
				ch.clear();
			}
			eachStep = 0;
		}
	}
	model::rs->exportEachResult2TXT(model::path);
	sendProgress(0, "__line__");
	QTime endingTime = tme.currentTime();
	QDate endingDate = QDate::currentDate();
	double dtime = tme.elapsed() * 0.001;
	int minute = static_cast<int>(dtime / 60.0);
	int hour = static_cast<int>(minute / 60.0);
	qts.setFieldWidth(0);
	int cgtime = endingTime.second() - startingTime.msec();
	qts << "     Starting time/date     = " << startingTime.toString() << " / " << startingDate.toString() << endl
		<< "     Ending time/date      = " << endingTime.toString() << " / " << endingDate.toString() << endl
		<< "     CPU + GPU time       = " << dtime << " second  ( " << hour << " h. " << minute << " m. " << dtime << " s. )";
	sendProgress(-1, ch); ch.clear();
	emit finishedThread();
}


