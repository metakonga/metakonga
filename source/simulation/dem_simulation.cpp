#include "dem_simulation.h"
#include "collision.h"
#include "polygonObject.h"
#include "object.h"
#include <iomanip>
#include "mphysics_cuda_dec.cuh"
#include <QFile>
#include <QDebug>
#include <QTime>
#include <QTextStream>

dem_simulation::dem_simulation()
	:simulation()
{

}

dem_simulation::dem_simulation(modeler *_md)
	:simulation(_md)
	, itor(NULL)
	, gb(NULL)
{
	
}

dem_simulation::~dem_simulation()
{
	if (itor) delete itor; itor = NULL;
	if (gb) delete gb; gb = NULL;
}

bool dem_simulation::initialize(bool isCpu)
{
	_isWait = false;
	_isWaiting = false;
	_abort = false;
	_interrupt = false;
	//pBar = new QProgressBar;
	//durationTime = new QLineEdit;
	//durationTime->setFrame(false);
	nstep = static_cast<unsigned int>((et / dt) + 1);
	//QProgressBar *pBar;
	gb = new neighborhood_cell("detector", md);
	gb->setWorldOrigin(VEC3F(-1.0f, -1.0f, -1.0f));
	gb->setGridSize(VEC3UI(128, 128, 128));
	gb->setCellSize(md->particleSystem()->maxRadius() * 2.0f);

	qDebug() << "- Allocation of contact detection module ------------------ DONE";
	itor = new velocity_verlet(md);
	qDebug() << "- Allocation of integration module ------------------------ DONE";

	unsigned int s_np = 0;
	if (md->objPolygon().size()){
		s_np = md->numPolygonSphere();
	}

	if (isCpu){
		gb->allocMemory(md->numParticle() + s_np);
		foreach(object* value, md->objects())
		{
// 			if (value->rolltype() != ROLL_PARTICLE)
// 			{
// 				value->setRelativeImpactVelocity(md->numParticle());
// 			}
		}
	}
	else{
		
		gb->cuAllocMemory(md->numParticle() + s_np);
		md->particleSystem()->cuAllocMemory();
	//	double maxRad = (double)(md->particleSystem()->maxRadius());
		foreach(object* value, md->objects())
		{
			if (value->rolltype() != ROLL_PARTICLE)
			{
				value->cuAllocData(md->numParticle());
			}
// 			if (value->objectType() == POLYGON)
// 			{
// 				polygonObject* po = dynamic_cast<polygonObject*>(value);
// 				if (maxRad < po->maxRadius())
// 					maxRad = po->maxRadius();
// 			}
		}
	//	gb->setCellSize((float)(maxRad * 2.0f));
		
		device_parameters paras;
		paras.np = md->numParticle();
		paras.nsphere = s_np;
		paras.dt = dt;
		paras.cohesion = 0.0f;// 1.0E+6;
		paras.half2dt = 0.5f * dt * dt;
		paras.gravity = make_float3(md->gravity().x, md->gravity().y, md->gravity().z);
		paras.cell_size = grid_base::cs;
		paras.ncell = gb->nCell();
		paras.grid_size = make_uint3(grid_base::gs.x, grid_base::gs.y, grid_base::gs.z);
		paras.world_origin = make_float3(grid_base::wo.x, grid_base::wo.y, grid_base::wo.z);
		setSymbolicParameter(&paras);
	}
	//gb->reorderElements(isCpu);
	foreach(collision* value, md->collisions())
	{
		if (value->getCollisionPairType() == PARTICLES_POLYGONOBJECT){
			value->setGridBase(gb);
		}
	}

	//md->particleSystem()->velocity()[0].x = 0.1f;
	
	return true;
}

bool dem_simulation::saveResult(float ct, unsigned int p)
{
	char partName[256] = { 0, };
	//double radius = 0.0;
	sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", md->modelPath().toStdString().c_str(), p);
	//std::fstream of;
	QFile of(partName);
	unsigned int np = md->numParticle();
	//of.open(partName, std::ios::out, std::ios::binary);
	of.open(QIODevice::WriteOnly);
	of.write((char*)&np, sizeof(unsigned int));
	of.write((char*)&ct, sizeof(float));
	of.write((char*)md->particleSystem()->position(), sizeof(VEC4F) * md->numParticle());
	of.write((char*)md->particleSystem()->velocity(), sizeof(VEC3F) * md->numParticle());
	of.close();
	return true;
}

bool dem_simulation::cuSaveResult(float ct, unsigned int p)
{
	char partName[256] = { 0, };
	//double radius = 0.0;
	sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", md->modelPath().toStdString().c_str(), p);
	//std::fstream of;
	QFile of(partName);
	unsigned int np = md->numParticle();
	checkCudaErrors(cudaMemcpy(md->particleSystem()->position(), md->particleSystem()->cuPosition(), sizeof(float)*md->numParticle() * 4, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(md->particleSystem()->velocity(), md->particleSystem()->cuVelocity(), sizeof(float)*md->numParticle() * 3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(md->particleSystem()->force(), md->particleSystem()->cuForce(), sizeof(float)*md->numParticle() * 3, cudaMemcpyDeviceToHost));
	of.open(QIODevice::WriteOnly);
	of.write((char*)&np, sizeof(unsigned int));
	of.write((char*)&ct, sizeof(float));
	of.write((char*)md->particleSystem()->position(), sizeof(VEC4F) * md->numParticle());
	of.write((char*)md->particleSystem()->velocity(), sizeof(VEC3F) * md->numParticle());
	of.write((char*)md->particleSystem()->force(), sizeof(VEC3F) * md->numParticle());
	of.close();
	return true;
}

void dem_simulation::collision_dem(float dt)
{
	md->particleSystem()->particleCollision(dt);

// 	std::map<std::string, collision*>::iterator c;
// 	for (c = md->collision_map().begin(); c != md->collision_map().end(); c++){
// 		c->second->collid(dt);
// 	}
}

void dem_simulation::cuCollision_dem()
{
	md->particleSystem()->cuParticleCollision(gb);
	foreach(collision* value, md->collisions())
	{
		value->cuCollid();
	}

}

bool dem_simulation::cpuRun()
{
	unsigned int part = 0;
	unsigned int cStep = 0;
	unsigned int eachStep = 0;
	
	float ct = dt * cStep;
	qDebug() << "-------------------------------------------------------------" << endl
			 << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |" << endl
			 << "-------------------------------------------------------------";
	QTextStream::AlignRight;
	//QTextStream::setRealNumberPrecision(6);
	QTextStream os(stdout);
	os.setRealNumberPrecision(6);
	if (saveResult(ct, part)){
		os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << "0"  << qSetFieldWidth(0) << " |" << endl;
		//std::cout << "| " << std::setw(9) << part << std::setw(12) << ct << std::setw(10) << eachStep << std::setw(11) << cStep << std::setw(15) << 0 << std::endl;
	}
	QTime tme;
	tme.start();
	cStep++;	
	//md->particleSystem()->velocity()[0].x = 1.0f; //initial particles velocity setting 
	while (cStep < nstep)
	{
		if (_abort){
			_interrupt = true;
			return false;
		}
		if (_isWait){
			_isWaiting = true;
			continue;
		}
		//mutex.lock();
		ct = dt * cStep;
// 		if (cStep == 20000){
// 			cStep = 20000;
// 		}
		itor->updatePosition(dt);
// 		for (unsigned int i = 0; i < 8; i++)
// 			ppf << ct << " " << md->particleSystem()->position()[i].x << " " << md->particleSystem()->position()[i].y << " " << md->particleSystem()->position()[i].z;
		gb->detection();
		collision_dem(dt);
// 		for (unsigned int i = 0; i < 8; i++)
// 			ppf << md->particleSystem()->force()[i].x << " " << md->particleSystem()->force()[i].y << " " << md->particleSystem()->force()[i].z;
// 		ppf << endl;
		itor->updateVelocity(dt);
		//md->updateObject(dt);
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
		//mutex.unlock();
	}
//	pf.close();
	emit finished();
	return true;
}

bool dem_simulation::gpuRun()
{


	unsigned int part = 0;
	unsigned int cStep = 0;
	unsigned int eachStep = 0;
	ct = dt * cStep;
	qDebug() << "-------------------------------------------------------------" << endl
			 << "| Num. Part | Sim. Time | I. Part | I. Total | Elapsed Time |" << endl
			 << "-------------------------------------------------------------";
	QTextStream::AlignRight;
	QTextStream os(stdout);
	os.setRealNumberPrecision(6);
	if (cuSaveResult(ct, part)){
		os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << "0" << qSetFieldWidth(0) << " |" << endl;
	}
	QTime tme;
	tme.start();
	cStep++;
	while (cStep < nstep)
	{
		if (_isWait)
			continue;
		if (_abort){
			emit finished();
			return false;
		}
		ct = dt * cStep;
		//std::cout << "step 1" << std::endl;
		itor->cuUpdatePosition();
		//std::cout << "step 2" << std::endl;
 		gb->cuDetection();
		//std::cout << "step 3" << std::endl;
 		cuCollision_dem();
		//std::cout << "step 4" << std::endl;
 		itor->cuUpdateVelocity();
		//std::cout << "step 5" << std::endl;
	//	md->updateObject(dt, GPU);
		//std::cout << "step 6" << std::endl;
		if (!((cStep) % step)){
			part++;
			emit sendProgress(part);
			if (cuSaveResult(ct, part)){
				os << "| " << qSetFieldWidth(9) << part << qSetFieldWidth(12) << ct << qSetFieldWidth(10) << eachStep << qSetFieldWidth(11) << cStep << qSetFieldWidth(15) << tme.elapsed() * 0.001 << qSetFieldWidth(0) << " |" << endl;
			}
			eachStep = 0;
		}
		cStep++;
		eachStep++;
	}
	emit finished();
	return true;
}