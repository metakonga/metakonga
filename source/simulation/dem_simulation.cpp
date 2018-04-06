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
	, itor(NULL), gb(NULL), paras(NULL)
	, d_pos(NULL), d_vel(NULL), d_acc(NULL)
	, d_omega(NULL), d_alpha(NULL), d_fr(NULL)
	, d_mm(NULL), d_ms(NULL), d_iner(NULL)
	, d_riv(NULL)
{

}

dem_simulation::dem_simulation(modeler *_md)
	:simulation(_md)
	, itor(NULL), gb(NULL), paras(NULL)
	, d_pos(NULL), d_vel(NULL), d_acc(NULL)
	, d_omega(NULL), d_alpha(NULL), d_fr(NULL)
	, d_mm(NULL), d_ms(NULL), d_iner(NULL)
	, d_riv(NULL)
{
	
}

dem_simulation::~dem_simulation()
{
	clear();
}

void dem_simulation::clear()
{
	if (itor) delete itor; itor = NULL;
	if (gb) delete gb; gb = NULL;
	if (paras) delete paras; paras = NULL;

	if (d_pos) checkCudaErrors(cudaFree(d_pos)); d_pos = NULL;
	if (d_vel) checkCudaErrors(cudaFree(d_vel)); d_vel = NULL;
	if (d_acc) checkCudaErrors(cudaFree(d_acc)); d_acc = NULL;
	if (d_omega) checkCudaErrors(cudaFree(d_omega)); d_omega = NULL;
	if (d_alpha) checkCudaErrors(cudaFree(d_alpha)); d_alpha = NULL;
	if (d_fr) checkCudaErrors(cudaFree(d_fr)); d_fr = NULL;
	if (d_mm) checkCudaErrors(cudaFree(d_mm)); d_mm = NULL;
	if (d_ms) checkCudaErrors(cudaFree(d_ms)); d_ms = NULL;
	//if (d_rad) checkCudaErrors(cudaFree(d_rad));
	if (d_iner) checkCudaErrors(cudaFree(d_iner)); d_iner = NULL;
	if (d_riv) checkCudaErrors(cudaFree(d_riv)); d_riv = NULL;
}

bool dem_simulation::initialize(bool isCpu)
{
	clear();
	_isCpu = isCpu;
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
	gb->setWorldOrigin(VEC3D(-1.0, -1.0, -1.0));
	gb->setGridSize(VEC3UI(128, 128, 128));
	gb->setCellSize(md->particleSystem()->maxRadius() * 2.0);

	qDebug() << "- Allocation of contact detection module ------------------ DONE";
	itor = new velocity_verlet(md);
	qDebug() << "- Allocation of integration module ------------------------ DONE";

	unsigned int s_np = 0;
	if (md->numPoly()){
		s_np = md->numPolygonSphere();
	}
	np = md->numParticle();
	m_pos = new VEC4D[np];
	m_vel = new VEC3D[np];
	m_force = new VEC3D[np];
	if (isCpu){
		gb->allocMemory(np);
		memcpy(m_pos, md->particleSystem()->position(), sizeof(double) * 4 * np);
		memcpy(m_vel, md->particleSystem()->velocity(), sizeof(double) * 3 * np);
	}
	else{
		gb->cuAllocMemory(np);
		//md->particleSystem()->cuAllocMemory();
		cudaAllocMemory(np);
		foreach(object* value, md->objects())
		{
			if (value->rolltype() != ROLL_PARTICLE)
			{
				value->cuAllocData(md->numParticle());
			}
		}
		foreach(collision* value, md->collisions())
		{
			value->allocDeviceMemory();
		}
		//device_parameters paras;
		paras = new device_parameters;
		if (md->particleSystem()->generationMethod()){
			paras->np = md->particleSystem()->numParticlePerStack();
		}
		else{
			paras->np = md->numParticle();
		}
		//paras->np = ->particleSystem()->numParticlePerStack();// md->numParticle();
		paras->nsphere = s_np;
		paras->dt = dt;
		paras->cohesion = 0.0;// 1.0E+6;
		paras->half2dt = 0.5 * dt * dt;
		paras->gravity = make_double3(md->gravity().x, md->gravity().y, md->gravity().z);
		paras->cell_size = grid_base::cs;
		paras->ncell = gb->nCell();
		paras->grid_size = make_uint3(grid_base::gs.x, grid_base::gs.y, grid_base::gs.z);
		paras->world_origin = make_double3(grid_base::wo.x, grid_base::wo.y, grid_base::wo.z);
		setSymbolicParameter(paras);
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

bool dem_simulation::saveResult(double ct, unsigned int p)
{
	char partName[256] = { 0, };
	//double radius = 0.0;
	sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", md->modelPath().toStdString().c_str(), p);
	//std::fstream of;
	QFile of(partName);
	//unsigned int np = md->numParticle();
	//of.open(partName, std::ios::out, std::ios::binary);
	of.open(QIODevice::WriteOnly);
	of.write((char*)&np, sizeof(unsigned int));
	of.write((char*)&ct, sizeof(double));
	of.write((char*)m_pos, sizeof(VEC4D) * np);
	of.write((char*)m_vel, sizeof(VEC3D) * np);
	of.write((char*)m_force, sizeof(VEC3D) * np);
	of.close();
	return true;
}

bool dem_simulation::savePartResult(double ct, unsigned int p)
{
	char partName[256] = { 0, };
	//double radius = 0.0;
	sprintf_s(partName, sizeof(char) * 256, "%s/part%04d.bin", md->modelPath().toStdString().c_str(), p);
	//std::fstream of;
	QFile of(partName);
//	unsigned int np = md->numParticle();
	//unsigned int snp = md->particleSystem()->numStackParticle();
	if (!_isCpu)
	{
		checkCudaErrors(cudaMemcpy(m_pos, d_pos, sizeof(double)*np * 4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(m_vel, d_vel, sizeof(double)*np * 3, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(m_force, d_acc, sizeof(double)*np * 3, cudaMemcpyDeviceToHost));
	}	
	of.open(QIODevice::WriteOnly);
	of.write((char*)&np, sizeof(unsigned int));
	//of.write((char*)&snp, sizeof(unsigned int));
	of.write((char*)&ct, sizeof(double));
	of.write((char*)m_pos, sizeof(VEC4D) * np);
	of.write((char*)m_vel, sizeof(VEC3D) * np);
	of.write((char*)m_force, sizeof(VEC3D) * np);
	of.close();
	return true;
}

void dem_simulation::cudaUpdatePosition()
{
	itor->cuUpdatePosition(d_pos, d_vel, d_acc, np);
}

void dem_simulation::cudaDetection()
{
	gb->cuDetection(d_pos);
}

void dem_simulation::cudaUpdateVelocity()
{
	itor->cuUpdateVelocity(d_vel, d_acc, d_omega, d_alpha, d_fr, d_mm, d_ms, d_iner, np);
}

void dem_simulation::collision_dem(double dt)
{
//	md->particleSystem()->particleCollision(dt);

// 	std::map<std::string, collision*>::iterator c;
// 	for (c = md->collision_map().begin(); c != md->collision_map().end(); c++){
// 		c->second->collid(dt);
// 	}
	foreach(collision* value, md->collisions())
	{
		value->collid(dt);
	} 
}

void dem_simulation::cuCollision_dem()
{
	//md->particleSystem()->cuParticleCollision();
	foreach(collision* value, md->collisions())
	{
		value->cuCollid(d_pos, d_vel, d_omega, d_ms, d_fr, d_mm, np);
	}

}

void dem_simulation::cudaAllocMemory(unsigned int np)
{
	checkCudaErrors(cudaMalloc((void**)&d_pos, sizeof(double)*np * 4));
	checkCudaErrors(cudaMemcpy(d_pos, md->particleSystem()->position(), sizeof(double) * np * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_vel, sizeof(double)*np * 3));
	//vel[0].z = 0.1f;
	checkCudaErrors(cudaMemcpy(d_vel, md->particleSystem()->velocity(), sizeof(double) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_acc, sizeof(double)*np * 3));
	checkCudaErrors(cudaMemcpy(d_acc, md->particleSystem()->acceleration(), sizeof(double) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_omega, sizeof(double)*np * 3));
	checkCudaErrors(cudaMemset(d_omega, 0, sizeof(double) * np * 3));
//	checkCudaErrors(cudaMemcpy(d_omega, omega, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_alpha, sizeof(double)*np * 3));
	checkCudaErrors(cudaMemset(d_alpha, 0, sizeof(double) * np * 3));
	//checkCudaErrors(cudaMemcpy(d_alpha, alpha, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_fr, sizeof(double)*np * 3));
	checkCudaErrors(cudaMemcpy(d_fr, md->particleSystem()->force() , sizeof(double) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_mm, sizeof(double)*np * 3));
	checkCudaErrors(cudaMemset(d_mm, 0, sizeof(double) * np * 3));
	//checkCudaErrors(cudaMemcpy(d_mm, mm, sizeof(double) * np * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_ms, sizeof(double)*np));
	checkCudaErrors(cudaMemcpy(d_ms, md->particleSystem()->mass(), sizeof(double) * np, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMalloc((void**)&d_rad, sizeof(double)*np));
	//checkCudaErrors(cudaMemcpy(d_rad, rad, sizeof(double) * np, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_iner, sizeof(double)*np));
	checkCudaErrors(cudaMemcpy(d_iner, md->particleSystem()->inertia(), sizeof(double) * np, cudaMemcpyHostToDevice));
}

bool dem_simulation::cpuRun()
{
	unsigned int part = 0;
	unsigned int cStep = 0;
	unsigned int eachStep = 0;
	
	double ct = dt * cStep;
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
// 	std::fstream fs;
// 	fs.open("C:/C++/presult.txt", std::ios::out);
	//md->particleSystem()->velocity()[0].x = 1.0f; //initial particles velocity setting 
	while (cStep < nstep)
	{
		std::cout << cStep << std::endl;
		if (cStep == 31617)
			bool p = true;
		if (!(cStep % 2000))
			md->particleSystem()->appendCluster();

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
		
		md->particleSystem()->clusterUpdatePosition(dt);
		//itor->updatePosition(dt);
// 		for (unsigned int i = 0; i < 8; i++)
// 			ppf << ct << " " << md->particleSystem()->position()[i].x << " " << md->particleSystem()->position()[i].y << " " << md->particleSystem()->position()[i].z;
		gb->detection();
		collision_dem(dt);
// 		for (unsigned int i = 0; i < 8; i++)
// 			ppf << md->particleSystem()->force()[i].x << " " << md->particleSystem()->force()[i].y << " " << md->particleSystem()->force()[i].z;
// 		ppf << endl;
		md->particleSystem()->clusterUpdateVelocity(dt);
		//itor->updateVelocity(dt);
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
	}
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
	if (savePartResult(ct, part)){
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
// 		if (md->particleSystem()->updateStackParticle(ct)){
// 			paras->np += md->particleSystem()->numParticlePerStack();// ->numParticle();
// 			//gb->cuResizeMemory(paras.np);
// 			setSymbolicParameter(paras);
// 		}
		itor->updatePosition(d_pos, d_vel, d_acc, np);
 		gb->cuDetection(d_pos);
 		cuCollision_dem();
		itor->updateVelocity(d_vel, d_acc, d_omega, d_alpha, d_fr, d_mm, d_ms, d_iner, np);
		if (!((cStep) % step)){
			part++;
			emit sendProgress(part);
			if (savePartResult(ct, part)){
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