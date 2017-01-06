#include "parSIM.h"
#include "timer.h"
#include <ctime>
#include <direct.h>
#include <iomanip>

#include <lapack/f2c.h>
#include <lapack/clapack.h>

using namespace parSIM;

Mbdsimulation::Mbdsimulation(Simulation* sim)
	: mforce(NULL)
// 	, pos(NULL)
// 	, vel(NULL)
// 	, acc(NULL)
// 	, ep(NULL)
// 	, ev(NULL)
// 	, ea(NULL)
	, lhs(NULL)
	, rhs(NULL)
	, cjaco(NULL)
	///, permutation(NULL)
	, nmass(0)
{
	if(sim){
		Simulation::base_path = sim->getBasePath();
		Simulation::Name = sim->getName();
		Simulation::CaseName = sim->getCaseName();
		Simulation::Device = sim->getDeviceType();
		Simulation::Dim = sim->getDim();
		Simulation::Float_type = sim->getPrecision();
		Simulation::save_dt = sim->getSaveDt();
		Simulation::sim_time = sim->getSimTime();
		Simulation::Dt = sim->getDt();

		Simulation::geometries = sim->getGeometries();
		Simulation::ps = sim->getParticles();

		sim->setSubSimulation(this);
	}	
}

Mbdsimulation::~Mbdsimulation()
{
	clear();
}

void Mbdsimulation::clear()
{
	if(lhs) delete lhs; lhs = NULL;
	if(rhs) delete rhs; rhs = NULL;
	if(cjaco) delete cjaco; cjaco = NULL;
	if(mforce) delete mforce; mforce = NULL;
}

void Mbdsimulation::cpu_run()
{

}

void Mbdsimulation::gpu_run()
{

}

bool Mbdsimulation::initialize_simulation()
{
	clear();
	nmass = masses->size();
	lhs = new matrix<double>(nmass * 8, nmass * 8);
	rhs = new algebra::vector<double>(nmass * 8);

	unsigned int i = 0;
	for(pm = masses->begin(); pm != masses->end(); pm++, i++){
		pm->second->setInertia();
		pm->second->Velocity() += vector3<double>(10.0, 0.0 ,0.0);
		std::map<std::string, geometry*>::iterator Geo = geometries->find(pm->first);
		if(Geo == geometries->end())
			continue;
		Geo->second->bindPointMass(pm->second);
		if(Device == GPU){
			pm->second->define_device_info();
		}
	}

	if(!mforce)
		mforce = new mbd_force(this);
	mforce->setGavity(0.0, -9.80665, 0.0);
	unsigned int mDim = masses->size() * 7;
	unsigned int tDim = masses->size() * 8;
	integer lapack_one = 1;
	integer lapack_info = 0;
	integer *ptDof = (integer*)&tDim;
	integer *permutation = new integer[*ptDof];
	cjaco = new sparse_matrix<double>(masses->size() * 4, tDim - mDim, mDim);

	calculateMassMatrix();
	calculateForceVector();
	//lhs->display();
	i = 0;
	dgesv_(ptDof, &lapack_one, lhs->getDataPointer(), ptDof, permutation, rhs->get_ptr(), ptDof, &lapack_info);
	for(pm = masses->begin(); pm != masses->end(); pm++, i++){
		pointmass* m = pm->second;
		m->Acceleration() = vector3<double>((*rhs)(i), (*rhs)(i+1), (*rhs)(i+2));
	}
	delete [] permutation; permutation = NULL;
	
	return true;
}

void Mbdsimulation::calculateMassMatrix(double mul)
{
	int cnt = 0;
	lhs->zeros();
	unsigned int i = 0, sRow = cjaco->rows() - nmass + nmass * 7;
	matrix4x4<double> LTJL;
	matrix3x4<double> Gp;
	cjaco->zeroCount();
	for(pm = masses->begin(); pm != masses->end(); pm++){
		pointmass* m = pm->second;
		(*lhs)(cnt, cnt) = (*lhs)(cnt + 1, cnt + 1) = (*lhs)(cnt + 2, cnt + 2) = mul * m->Mass();
		i = cnt + 3;
		Gp = G(m->Orientation());
		LTJL = mul * 4 * transpose(Gp, m->Inertia() * Gp);
		for(int j(0); j < 4; j++){
			for(int k(0); k < 4; k++){
				if(LTJL(j,k)) 
					(*lhs)(i+j, i+k) = LTJL(j,k);
			}
		}
		vector4<double> tep = 2 * m->Orientation();
		cjaco->extraction(sRow++, i, POINTER4(tep), 4);
	}
	for(int i(0); i < cjaco->nnz(); i++){
		(*lhs)(cjaco->ridx[i], cjaco->cidx[i]) = (*lhs)(cjaco->cidx[i], cjaco->ridx[i]) = cjaco->value[i];
	}
}

void Mbdsimulation::calculateForceVector()
{
	rhs->zeros();
	int cnt = 0;
	vector3d nf = vector3d(0.0);
	vector4d rf = vector4d(0.0);
	for(pm = masses->begin(); pm != masses->end(); pm++){
		pointmass* m = pm->second;
		nf = m->Mass() * mforce->Gravity();
		geo::shape *sh = dynamic_cast<geo::shape*>(geometries->find(m->Name())->second);
		nf += sh->body_force;
		rf = calculateInertiaForce(m->dOrientation(), m->Inertia(), m->Orientation());
		rhs->insert(cnt, POINTER3(nf), POINTER4(rf), 3, 4);
		cnt+=7;
	}
}

vector4<double> Mbdsimulation::calculateInertiaForce(vector4<double>& ev, matrix3x3<double>& J, vector4<double>& ep)
{
	return vector4<double>(
		8*(ev.z*(J.a11*ep.x*ev.y - J.a00*ep.y*ev.x + J.a22*ep.w*ev.z) + ev.y*(J.a11*ep.w*ev.y + J.a00*ep.z*ev.x - J.a22*ep.x*ev.z) + ev.x*(J.a00*ep.w*ev.x - J.a11*ep.z*ev.y + J.a22*ep.y*ev.z) - ev.w*(J.a00*ep.x*ev.x + J.a11*ep.y*ev.y + J.a22*ep.z*ev.z)),
		8*(ev.z*(J.a00*ep.y*ev.w + J.a11*ep.x*ev.z - J.a22*ep.w*ev.y) + ev.y*(J.a11*ep.w*ev.z - J.a00*ep.z*ev.w + J.a22*ep.x*ev.y) - ev.x*(J.a00*ep.w*ev.w + J.a11*ep.z*ev.z + J.a22*ep.y*ev.y) + ev.w*(J.a00*ep.x*ev.w - J.a11*ep.y*ev.z + J.a22*ep.z*ev.y)),
		8*(ev.z*(J.a00*ep.y*ev.z - J.a11*ep.x*ev.w + J.a22*ep.w*ev.x) - ev.y*(J.a11*ep.w*ev.w + J.a00*ep.z*ev.z + J.a22*ep.x*ev.x) + ev.x*(J.a11*ep.z*ev.w - J.a00*ep.w*ev.z + J.a22*ep.y*ev.x) + ev.w*(J.a00*ep.x*ev.z + J.a11*ep.y*ev.w - J.a22*ep.z*ev.x)),
		8*(ev.y*(J.a00*ep.z*ev.y - J.a11*ep.w*ev.x + J.a22*ep.x*ev.w) - ev.z*(J.a00*ep.y*ev.y + J.a11*ep.x*ev.x + J.a22*ep.w*ev.w) + ev.x*(J.a00*ep.w*ev.y + J.a11*ep.z*ev.x - J.a22*ep.y*ev.w) + ev.w*(J.a11*ep.y*ev.x - J.a00*ep.x*ev.y + J.a22*ep.z*ev.w))
		);
}

void Mbdsimulation::oneStepAnalysis()
{
	unsigned int tDim = masses->size() * 8;
	integer lapack_one = 1;
	integer lapack_info = 0;
	integer *ptDof = (integer*)&tDim;
	integer *permutation = new integer[*ptDof];
//	cjaco = new sparse_matrix<double>(masses->size() * 4, tDim - mDim, mDim);

	calculateMassMatrix();
	calculateForceVector();
//	lhs->display();
	unsigned int i = 0;
	dgesv_(ptDof, &lapack_one, lhs->getDataPointer(), ptDof, permutation, rhs->get_ptr(), ptDof, &lapack_info);
	for(pm = masses->begin(); pm != masses->end(); pm++, i++){
		pointmass* m = pm->second;
		m->Acceleration() = vector3<double>((*rhs)(i), (*rhs)(i+1), (*rhs)(i+2));
	}
	/*checkCudaErrors( cudaMemcpy((void**)&m))*/
	delete [] permutation; permutation = NULL;
}

bool Mbdsimulation::RunSim()
{
	return true;
}