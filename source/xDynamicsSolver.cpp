#include "xDynamicsSolver.h"
#include "modelManager.h"
#include "messageBox.h"
//#include "errors.h"
#include "startingModel.h"
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

bool xDynamicsSolver::initialize(int dem_itype, int mbd_itype, startingModel* stm)
{	
	nstep = static_cast<unsigned int>((simulation::et / simulation::dt) + 1);
	npart = static_cast<unsigned int>((nstep / simulation::st) + 1);
	if (dem)
	{
		dem->setIntegratorType((dem_integrator_type)dem_itype);
		particleManager* pm = mg->DEMModel()->ParticleManager();
		if (stm && pm->RealTimeCreating())
			pm->setRealTimeCreating(false);

		unsigned int np = mg->DEMModel()->ParticleManager()->Np();
		if (model::isSinglePrecision)
			dem->initialize_f(mg->ContactManager());
		else
			dem->initialize(mg->ContactManager());
		if (stm)
			dem->setStartingData(stm);
			
		model::rs->setResultMemoryDEM(model::isSinglePrecision, npart, np);
		double m_size = model::rs->RequriedMemory(np, npart, DEM) / 1000000.0;
		sendProgress(-1, "Memory size of the result storage is " + QString("%1").arg(m_size) + "(MB)");
	}
	if (mbd)
	{
		mbd->setIntegratorType((mbd_integrator_type)mbd_itype);
		mbd->initialize(stm);
	}
	foreach(object* o, mg->GeometryObject()->Objects())
	{
		if (o->MotionCondition().enable)
			gms[o->Name()] = o;
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
		QString part_name;
		if (model::isSinglePrecision)
		{
			float* v_pos = model::rs->getPartPosition_f(pt);
			part_name = dem->saveResult_f(v_pos, NULL, ct, pt);
		}
		else
		{
			double *v_pos = model::rs->getPartPosition(pt);
			//double *v_vel = model::rs->getPartVelocity(pt);
			part_name = dem->saveResult(v_pos, NULL, ct, pt);
		}
		model::rs->insertPartName(part_name);
		//model::rs->definePartDatasDEM(false, pt);
	}
	if (mbd)
	{
		mbd->saveResult(ct);
	}
	foreach(object* o, mg->GeometryObject()->Objects())
	{
		pointMass* pm = dynamic_cast<pointMass*>(o);
		geometry_motion_result gmr = { pm->Position(), pm->getEP() };
		model::rs->insertGeometryObjectResult(o->Name(), gmr);
	}
	return true;
}

bool xDynamicsSolver::saveFinalResult(double ct)
{
	QString file = model::path + "/" + model::name + "_final.bfr";
	QFile qf(file);
	qf.open(QIODevice::WriteOnly);
	qf.write((char*)&ct, sizeof(double));
	if (dem)
	{
		model::isSinglePrecision ?
			dem->saveFinalResult_f(qf) :
			dem->saveFinalResult(qf);
	}
// 	if (mbd)
// 	{
// 		mbd->saveFinalResult(qf);
// 	}
	qf.close();
		//dem->saveFinalResult(file);
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
	
	QTime tme;
	//savePart(ct, part);
	tme.start();
	QTime startingTime = tme.currentTime();
	QDate startingDate = QDate::currentDate();
	int mbd_state = 0;
	while (cstep < nstep)
	{
		QMutexLocker locker(&m_mutex);
		if (isStop)
			break;
		cstep++;
		eachStep++;
		ct += simulation::dt;
		qDebug() << ct;

		simulation::setCurrentTime(ct);
		if (gms.size())
		{
			foreach(object* o, gms)
			{
				o->UpdateGeometryMotion(simulation::dt);
			}
		}
		if (dem)
		{
			model::isSinglePrecision ?
				dem->oneStepAnalysis_f(ct, cstep) :
				dem->oneStepAnalysis(ct, cstep);
			//qDebug() << "dem done";
		}
			
		if (mbd)
		{
			mbd_state = mbd->oneStepAnalysis(ct, cstep);
			//qDebug() << "mbd_state : " << mbd_state << "NR_Iteration : " << mbd->N_NR_Iteration();
			if (mbd_state == -1)
			{
				//errors::Error(mbd->MbdModel()->modelName());
				//break;
			}
			else if (mbd_state == 1)
			{
				if (mg->ContactManager())
					mg->ContactManager()->update();// ContactParticlesPolygonObjects()->updatePolygonObjectData();
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
	saveFinalResult(ct);
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


