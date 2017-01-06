#include "parSIM.h"

#ifndef F2C_INCLUDE
#include <lapack/f2c.h>
#endif
#ifndef __CLAPACK_H
#include <lapack/clapack.h>
#endif

using namespace parSIM;

Mbdsimulation::Mbdsimulation(std::string name)
	: time(0)
	, mDim(0)
	, tDim(0)
	, Dof(0)
	, ptDof(NULL)
	, cjacoNNZ(NULL)
	, Simulation(name)
{
	for(int i = 0; i < NUM_INTEGRATOR; i++){
		itor[i] = NULL;
	}
}

Mbdsimulation::~Mbdsimulation()
{
	for(int i = 0; i < NUM_INTEGRATOR; i++){
		if(itor[i])
			delete itor[i];
	}
}

bool Mbdsimulation::initialize()
{
	int sDim = 0;
	lapack_one = 1;
	lapack_info = 0;
	alpha = -0.3;
	beta = (1 - alpha) * (1 - alpha) / 4;
	gamma = 0.5 - alpha;
	eps=1E-6;
	
	switch(integrator){
	case IMPLICIT_HHT: itor[integrator] = new HHT_integrator(this); break;
	}
	//itor[integrator]->setDt(Simulation::dt);

	if(masses.size())
	{
		mDim = (masses.size() - 1) * 7;
	}

	for(KConstraintIterator it = kinConsts.begin(); it != kinConsts.end(); it++){
		kinematicConstraint *kconst = it->second;
		kconst->sRow = sDim;
		kconst->iCol = (kconst->i - 1) * 7;
		kconst->jCol = (kconst->j - 1) * 7;
		switch (kconst->type)
		{
		case kinematicConstraint::REVOLUTE:				sDim+=5; break;
		case kinematicConstraint::CYLINDERICAL:			sDim+=4; break;
		case kinematicConstraint::SPHERICAL:			sDim+=3; break;
		case kinematicConstraint::TRANSLATION:			sDim+=5; break;
		case kinematicConstraint::UNIVERSAL:			sDim+=4; break;
		case kinematicConstraint::DRIVING_CYLINDERICAL:	sDim+=2; break;
		default:
			break;
		}
		cjacoNNZ += kconst->max_nnz;
	}
	Dof = mDim - sDim;
	if(masses.size() == 0){
		std::cout << "Not exist mass element." << std::endl;
		return true;
	}
	sDim += masses.size() - 1;

	tDim = mDim + sDim;

	if(Simulation::specific_data != ""){
		std::fstream pf;
		pf.open(Simulation::specific_data, std::ios::in | std::ios::binary);
		if(pf.is_open()){
			while(!pf.eof()){
				int type;
				pf.read((char*)&type, sizeof(int));
				switch (type)
				{
				case -1:
					break;
				case SHAPE:{
					unsigned int name_size = 0;
					pf.read((char*)&name_size, sizeof(unsigned int));
					char cname[256] = {0, };
					pf.read(cname, sizeof(char)*name_size);
					std::string stdname = cname;
					std::map<std::string, pointmass*>::iterator mit = masses.find(stdname);
					float tv3[3] = {0, };
					pf.read((char*)tv3, sizeof(float)*3);
					mit->second->setPosition(vector3<double>(static_cast<double>(tv3[0]),
															 static_cast<double>(tv3[1]),
															 static_cast<double>(tv3[2])));

// 					mit->second->setPosito
// 					vector3<double> velocity(static_cast<double>(tv3[0]),
// 						static_cast<double>(tv3[1]),
// 						static_cast<double>(tv3[2])
// 						);
					//delete [] cname;
						   }
						   break;
				default:
					break;
				}
			}
		}
		else
		{
			Log::Send(Log::Error, "No exist specific_data. The path is " + Simulation::specific_data);
			return false;
		}
	}
	
	ptDof = (long int*)&tDim;
	permutation = new long int[tDim];
	lhs.alloc(tDim, tDim);
	rhs.alloc(tDim);
	previous.alloc(mDim);
	ipp.alloc(mDim);
	ipv.alloc(tDim);
	ee.alloc(tDim);
	constEQ.alloc(tDim - mDim);
	cjaco.alloc(cjacoNNZ + (masses.size() - 1) * 4, tDim - mDim, mDim);

	FULL_LEOM();

	dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof,	permutation, rhs.get_ptr(), ptDof, &lapack_info);
	int i = 0;
	for(MassIterator mit = masses.begin(); mit != masses.end(); mit++){
		pointmass* mass = mit->second;
		if(!mass->ID()) continue;
		mass->setAcceleration(rhs.get_ptr() + i * 7);
		mass->setddOrientation(rhs.get_ptr() + i * 7 + 3);
	}
	//KCONSTRAINT::iterator kconst = kconsts->begin();
 	lagMul = rhs.get_ptr() + mDim;
 	predictor.init(rhs.get_ptr(), rhs.sizes());

	return true;
}

void Mbdsimulation::setIntParameter()
{
	dt2accp = Simulation::dt*Simulation::dt*(1-2*beta)*0.5;
	dt2accv = Simulation::dt*(1 - gamma);
	dt2acc = Simulation::dt*Simulation::dt*beta;
	divalpha = 1 / (1+alpha);
	divbeta = -1 / (beta*Simulation::dt*Simulation::dt);
}

void Mbdsimulation::Prediction(unsigned int cStep)
{
	unsigned int i = 0;
	forceCalculator::calculateForceVector(previous, masses);
	constraintCalculator::sparseConstraintJacobian(mDim, cjaco, masses, kinConsts, driConsts);
	for(i=0; i < cjaco.nnz(); i++){
		previous(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mDim];
	}
	previous *= alpha / (1 + alpha);
	i = 0;
	for(MassIterator mit = masses.begin(); mit != masses.end(); mit++){
		pointmass* mass = mit->second;
		if(!mass->ID()) continue;
		if(pointmass::OnMoving){
			mass->RunMoving(Simulation::dt * cStep);
		}
		else{
			for(int j(0); j < 3; j++){
				ipp(i*7+j) = mass->Position()(j) + mass->Velocity()(j) * Simulation::dt + rhs(i*7+j) * dt2accp;
				ipv(i*7+j) = mass->Velocity()(j) + rhs(i*7+j) * dt2accv;
			}
			for(int j(0); j < 4; j++){
				ipp(i*7+3+j) = mass->Orientation()(j) + mass->dOrientation()(j) * Simulation::dt + rhs(i*7+3+j) * dt2accp;
				ipv(i*7+3+j) = mass->dOrientation()(j) + rhs(i*7+3+j) * dt2accv;
			}
		}
	}

	//predictor.apply(cStep);
	i = 0;
	for(MassIterator mit = masses.begin(); mit != masses.end(); mit++){
		pointmass* mass = mit->second;
		if(!mass->ID()) continue;
		
		if(pointmass::OnMoving){
		}
		else{
			for(int j(0); j < 3; j++){
				mass->Position()(j) = ipp(i*7+j) + rhs(i*7+j) * dt2acc;
				mass->Velocity()(j) = ipv(i*7+j) + rhs(i*7+j) * Simulation::dt * gamma;
			}
			for(int j(0); j < 4; j++){
				mass->Orientation()(j) = ipp(i*7+3+j) + rhs(i*7+3+j) * dt2acc;
				mass->dOrientation()(j) = ipv(i*7+3+j) + rhs(i*7+3+j) * Simulation::dt * gamma;
			}
		}
		mass->MakeTransformationMatrix();
		if(mass->Geometry())
			mass->cu_update_geometry_data();//Geometry()->cu_update_geometry(&(mass->TransformationMatrix().a00));
		i++;
	}
}

void Mbdsimulation::ShapeDataUpdate()
{
	for(MassIterator mit = masses.begin(); mit != masses.end(); mit++){
		pointmass* mass = mit->second;
		if(!mass->ID()) continue;
		mass->MakeTransformationMatrix();
		if(mass->Geometry())
			mass->cu_update_geometry_data();//Geometry()->cu_update_geometry(&(mass->TransformationMatrix().a00));
	}
}

void Mbdsimulation::Correction(unsigned int cStep)
{
	while(1){
		forceCalculator::calculateForceVector(ee, masses);
		constraintCalculator::sparseConstraintJacobian(mDim, cjaco, masses, kinConsts,driConsts);
		for(int i(0); i < cjaco.nnz(); i++){
			ee(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mDim];
		}
		massCalculator::calculateMassMatrix(lhs, masses, divalpha);
		for(unsigned int i = 1, k = 0; i < masses.size(); i++, k+=7){
			ee(k+0) += -lhs(k+0,k+0) * rhs(k+0);
			ee(k+1) += -lhs(k+1,k+1) * rhs(k+1);
			ee(k+2) += -lhs(k+2,k+2) * rhs(k+2);
			for(unsigned int j=3; j < 7; j++){
				ee(k+j) += -(lhs(k+j,k+3)*rhs(k+3) + lhs(k+j,k+4)*rhs(k+4) + lhs(k+j,k+5)*rhs(k+5) + lhs(k+j,k+6)*rhs(k+6));
			}  
		}
		massCalculator::calculateSystemJacobian(lhs, masses, divalpha*beta*Simulation::dt*Simulation::dt);
		constraintCalculator::calculateSystemJacobian(lhs, lagMul, masses, kinConsts, beta*Simulation::dt*Simulation::dt);
		for(int i(0); i < cjaco.nnz(); i++){
			lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
		}
		for(int i(0); i < mDim; i++) ee(i) -= previous(i);
		constraintCalculator::constraintEquation(ee.get_ptr() + mDim, masses, kinConsts, driConsts, tDim-mDim, cStep*Simulation::dt, divbeta);
		e_norm = ee.norm();
		dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof, permutation, ee.get_ptr(), ptDof, &lapack_info);
		rhs += ee;
		unsigned int i = 0;
		for(MassIterator mit = masses.begin(); mit != masses.end(); mit++){
			pointmass* mass = mit->second;
			if(!mass->ID()) continue;
			if(mass->IsMovingPart()){
			
			}
			else{
				for(unsigned int j(0); j < 3; j++){
					mass->Position()(j) = ipp(i*7+j) + rhs(i*7+j) * dt2acc;
					mass->Velocity()(j) = ipv(i*7+j) + rhs(i*7+j) * Simulation::dt * gamma;
				}
				for(unsigned int j(0); j < 4; j++){
					mass->Orientation()(j) = ipp(i*7+3+j) + rhs(i*7+3+j) * dt2acc;
					mass->dOrientation()(j) = ipv(i*7+3+j) + rhs(i*7+3+j) * Simulation::dt * gamma;
				}
			}
			mass->MakeTransformationMatrix();
			if(mass->Geometry())
				mass->cu_update_geometry_data();//Geometry()->cu_update_geometry(&(mass->TransformationMatrix().a00));
			i++;
		}
		//std::cout << e_norm << std::endl;
		if(e_norm <= 1e-5) break;
	}
}

void Mbdsimulation::parameterInitialize(double dt)
{
// 	bContactForce = false;
// 	lapack_one = 1;
// 	lapack_info = 0;
// 	forceCalculator::setGravity(pDoc->getGravity());
// 	alpha = -0.3;
// 	beta = (1 - alpha) * (1 - alpha) / 4;
// 	gamma = 0.5 - alpha;
// 	eps=1E-6;
// 	dt2accp = dt*dt*(1-2*beta)*0.5;
// 	dt2accv = dt*(1 - gamma);
// 	dt2acc = dt*dt*beta;
// 	divalpha = 1 / (1+alpha);
// 	divbeta = -1 / (beta*dt*dt);
// 	lagMul=NULL;
// 	e_norm = 0;
// 	masses=new natural_pointmass;
// 	masses->setData(pDoc->getPointMasses());
// 	kconsts = pDoc->getKinematicConstraints();
// 	dconsts = pDoc->getDrivingConstraint();
// 	if(pDoc->sizeAppliedForceElement()){
// 		forceCalculator::bindingTime(&time);
// 		forceCalculator::setAppliedForceElements(pDoc->getAppliedForceElements());
// 	}
// 	tDim = pDoc->getTotalDimension();// + ;
// 	mDim = (masses->size()-1)*7;
// 	sDof = pDoc->getSystemDegreeOfFreedom();
// 	ptDof = (integer*)&tDim;
// 	permutation = new integer[tDim];
// 	lhs.alloc(tDim, tDim);
// 	rhs.alloc(tDim);
// 	previous.alloc(mDim);
// 	ipp.alloc(mDim);
// 	ipv.alloc(mDim);
// 	ee.alloc(tDim);
// 	constEQ.alloc(tDim - mDim);
// 	cjaco.alloc(pDoc->getSparseConstraintJacobianMaxNNZ() + (masses->size() - 1)*4, tDim - mDim, mDim);
// 
// 	time = 0;
// 	FULL_LEOM(lhs, rhs, cjaco);
// 	//std::cout << lhs;
// 	dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof,	permutation, rhs.get_ptr(), ptDof, &lapack_info);
// 	masses->acc.insert(7, rhs.get_ptr(), mDim);
// 	//KCONSTRAINT::iterator kconst = kconsts->begin();
// 	lagMul = rhs.get_ptr() + mDim;
// 	predictor.init(rhs.get_ptr(), rhs.sizes());
// 	predictor.getTimeStep() = dt;
	//time = cFrame*dt;

}

bool Mbdsimulation::RK4(double dt, double et)
{
// 	lapack_one = 1;
// 	lapack_info = 0;
// 	masses=new natural_pointmass;
// 	masses->setData(Doc->getPointMasses());
// 	kconsts = Doc->getKinematicConstraints();
// 	dconsts = Doc->getDrivingConstraint();
// 	if(Doc->sizeAppliedForceElement()){
// 		forceCalculator::bindingTime(&time);
// 		forceCalculator::setAppliedForceElements(Doc->getAppliedForceElements());
// 	}
// 	tDim = Doc->getTotalDimension();// + ;
// 	mDim = (masses->size()-1)*7;
// 	sDof = Doc->getSystemDegreeOfFreedom();
// 	ptDof = (integer*)&tDim;
// 	permutation = new integer[tDim];
// 	matrix<double> kjaco(tDim - mDim, mDim); kjaco.zeros();
// 	//matrix<double> kdjaco(mDim, mDim)
// 	algebra::vector<double> kconst(tDim - mDim); kconst.zeros();
// 	algebra::vector<int> uID(sDof); uID.zeros();
// 	algebra::vector<int> vID(mDim-sDof); vID.zeros();
// 	integer *vPermutation = new integer[mDim-sDof];
// 	lhs.alloc(tDim, tDim);
// 	rhs.alloc(tDim);
// 	cjaco.alloc(Doc->getSparseConstraintJacobianMaxNNZ() + (masses->size() - 1)*4, tDim - mDim, mDim);
// 
// 	constraintCalculator::constraintJacobian(kjaco, masses, kconsts, dconsts);
// 	coordinatePartitioning(kjaco, uID);
// 	int cnt = 0;
// 	for(int i=0, j=0; i < mDim; i++){
// 		if(i == uID(j)){
// 			j++;
// 			cnt++;
// 			if(j==uID.sizes()) j--;
// 		}
// 		else{
// 			vID(i - cnt) = i;
// 		}
// 	}
// 	int tFrame = static_cast<int>(et/dt) + 1;
// 	int cFrame = 0;
// 	integer vDof = vID.sizes();
// 	integer uDof = uID.sizes();
// 	matrix<double> vlhs(vID.sizes(), vID.sizes());
// 	algebra::vector<double> vrhs(vID.sizes());
// 	algebra::vector<double> urhs(mDim);
// 	algebra::vector<double> f1(mDim);
// 	algebra::vector<double> f2(mDim);
// 	algebra::vector<double> f3(mDim);
// 	algebra::vector<double> f4(mDim);
// 	algebra::vector<double> y(uID.sizes()*2);
// 	algebra::vector<double> yy(uID.sizes()*2);
// 	time = 0;
// 	while(tFrame > cFrame){
// 		cnt = 0;
// 		double *pPos = masses->pos.get_ptr() + 7;
// 		// positionAnalysis
// 		while(1){		
// 			constraintJacobian(kjaco, masses, kconsts, dconsts);
// 			//lhs.exportDataToText();
// 			constraintEquation(kconst, masses, kconsts, dconsts, cFrame*dt);
// 			kconst.toMinus();
// 			for(int i=0; i < vDof; i++){
// 				vrhs(i) = kconst(vID(i));
// 				for(unsigned j=0; j < kjaco.rows(); j++){
// 					vlhs(j, i) = kjaco(j, vID(i));
// 				}
// 			}
// 			std::cout << vlhs << std::endl;
// 			std::cout << kjaco;
// 			dgesv_(&vDof, &lapack_one, vlhs.getDataPointer(), &vDof, vPermutation, vrhs.get_ptr(), &vDof, &lapack_info);
// 			for(int i(0); i < vDof; i++){
// 				pPos[vID(i)] += vrhs(i);
// 			}
// 			if(rhs.norm() < 10e-6) break;
// 		}
// 
// 		// velocityAnalysis
// 		double *pVel = masses->vel.get_ptr() + 7;
// 		constraintJacobian(kjaco, masses, kconsts, dconsts);
// 		for(int i=0; i < vDof; i++){
// 			for(unsigned j=0; j < kjaco.rows(); j++){
// 				vlhs(j, i) = kjaco(j, vID(i));
// 			}
// 		}
// 		double tval=0;
// 		for(unsigned i=0; i < vID.sizes(); i++){
// 			for(unsigned j=0; j < uID.sizes(); j++){
// 				tval += kjaco(i,uID(j))*pVel[uID(j)];
// 			}
// 			vrhs(i) = -tval;
// 			tval = 0;
// 		}
// 		dgesv_(&vDof, &lapack_one, vlhs.getDataPointer(), &vDof, vPermutation, vrhs.get_ptr(), &vDof, &lapack_info);
// 		for(int i(0); i < vDof; i++){
// 			pVel[vID(i)] = vrhs(i);
// 		}
// 
// 		// RK4
// 		for(unsigned i=0; i < uID.sizes(); i++){
// 			y(i) = pPos[uID(i)]; y(uDof + i) = pVel[uID(i)];
// 		}
// 		urhs.zeros();
// 		FULL_LEOM(lhs, rhs, cjaco);
// 		constraintJacobian2(kjaco, masses, kconsts, dconsts);
// 		for(int i(0); i < vDof; i++){
// 			for(int j(0); j < mDim; j++){
// 				rhs(i+mDim) -= kjaco(i,j) * masses->vel(7+j);
// 			}
// 		}
// 		dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof,	permutation, rhs.get_ptr(), ptDof, &lapack_info);
// 		for(int i=0; i < mDim; i++){
// 			f1(i) = rhs(i);
// 		}
// 		//masses->updatePositionAndVelocity(0.5*dt*);
// 	}
// 
 	return true;
}

bool Mbdsimulation::HHTI3(double dt, double et)
{
// 	int tFrame = static_cast<int>(et/dt) + 1;
// 	int	cFrame = 0;
// 	int nResultSave = 0;
// 	//time = cFrame*dt;
// 	parameterInitialize(dt, Doc);
// 	if(!Doc->isSaveDynResult()) Doc->initializeResultData(tFrame);
// 	std::cout << "duration time : " << cFrame*dt << std::endl;
// 	Doc->addDynResultData(masses);
// 	std::fstream fout;
// 	fout.open("C:/opengmap/result/bodyPosition.txt", std::ios::out);
// 	nResultSave++;
// 	cFrame++;
// 	//vector<double> rForce(tDim-mDim);
// 	//rForce.zeros();
// 	// iteration
// 	int nIteration=0;
// 
// 	while(tFrame > cFrame){
// 		time = cFrame*dt;
// 		//rForce.zeros();
// 		if(!(cFrame%100)) std::cout << "duration time : " << time << std::endl;
// 		forceCalculator::calculateForceVector(previous, masses);
// 		constraintCalculator::sparseConstraintJacobian(mDim, cjaco, masses, kconsts);
// 		for(int i(0); i < cjaco.nnz(); i++){
// 			previous(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mDim]; 
// 			//rForce(cjaco.cidx[i]) += cjaco.value[i] * lagMul[cjaco.ridx[i] - mDim];
// 		}
// 		double *pPos = masses->pos.get_ptr() + 21;
// // 		for(int i(0); i < tDim-mDim; i++)
// // 		{
// 		fout << pPos[0] << " " << pPos[1] << " " << pPos[2]; 
// //		}
// 		fout << std::endl;
// // 		std::cout << rForce;
// // 		std::cout << cjaco;
// 		//previous -= (alpha / (1 + alpha)) * rForce;
// 		previous *= alpha / (1 + alpha);
// 		for(int i(0); i < mDim; i++){
// 			ipp(i) = masses->pos(i+7) + masses->vel(i+7)*dt + rhs(i)*dt2accp;
// 			ipv(i) = masses->vel(i+7) + rhs(i)*dt2accv;
// 		}
// 
// 		predictor.apply(cFrame);
// 		for(int i(0); i < mDim; i++){
// 			masses->pos(i+7) = ipp(i) + rhs(i)*dt2acc;
// 			masses->vel(i+7) = ipv(i) + rhs(i)*dt*gamma;
// 		}
// 
// 		//masses->parameterNormalize();
// 		masses->updateTransformationMatrix();
// 
// 		while(1){
// 			forceCalculator::calculateForceVector(ee, masses);
// 			constraintCalculator::sparseConstraintJacobian(mDim, cjaco, masses, kconsts);
// 			for(int i(0); i < cjaco.nnz(); i++){
// 				ee(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mDim]; 
// 			}
// 			massCalculator::calculateMassMatrix(lhs, masses, divalpha);
// 			for(int i=1, k=0; i < masses->size(); i++, k+=7){
// 				ee(k+0) += -lhs(k+0,k+0) * rhs(k+0);
// 				ee(k+1) += -lhs(k+1,k+1) * rhs(k+1);
// 				ee(k+2) += -lhs(k+2,k+2) * rhs(k+2);
// 				for(int j=3; j < 7; j++){
// 					ee(k+j) += -(lhs(k+j,k+3)*rhs(k+3) + lhs(k+j,k+4)*rhs(k+4) + lhs(k+j,k+5)*rhs(k+5) + lhs(k+j,k+6)*rhs(k+6));
// 				} 
// 			}
// 			massCalculator::calculateSystemJacobian(lhs, masses, divalpha*beta*dt*dt);
// 			constraintCalculator::calculateSystemJacobian(lhs, lagMul, masses, kconsts, beta*dt*dt);
// 			for(int i(0); i < cjaco.nnz(); i++){
// 				lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
// 			}
// 			//std::cout << lhs;
// 			for(int i(0); i < mDim; i++) ee(i) -= previous(i);
//     			constraintCalculator::constraintEquation(ee.get_ptr() + mDim, masses, kconsts, dconsts, tDim - mDim, cFrame*dt, divbeta);
// 			e_norm = ee.norm();
// 			dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof,	permutation, ee.get_ptr(), ptDof, &lapack_info);
// 			
// 			rhs+=ee;
// 			for(int i(0); i < mDim; i++){
// 				masses->pos(i+7) = ipp(i) + rhs(i)*dt2acc;
// 				masses->vel(i+7) = ipv(i) + rhs(i)*dt*gamma;
// 			}
// 			//masses->parameterNormalize();
// 			masses->updateTransformationMatrix();
// 			if(e_norm <= 1e-5) break;
// 			nIteration++;
// 		}
// 		masses->acc.insert(7, rhs.get_ptr(), mDim);
// 		if(!(cFrame%10)){
// 			nResultSave++;
// 			Doc->addDynResultData(masses);
// 		}
// 		cFrame++;
// 		nIteration = 0;
// 	}
// 	//out.close();
// 	Doc->setAnimationTotalFrame(nResultSave);
// 	std::cout << "simulation done" << std::endl;
// 	fout.close();
// 	delete masses; masses = NULL;
 	return true;
}

void Mbdsimulation::FULL_LEOM()
{
 	massCalculator::calculateMassMatrix(lhs, masses);
 	forceCalculator::calculateForceVector(rhs, masses);
 	constraintCalculator::sparseConstraintJacobian((masses.size()-1)*7, cjaco, masses, kinConsts, driConsts);
	for(int i(0); i < cjaco.nnz(); i++){
		lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
	}
	//lhs.display();
}

void Mbdsimulation::calculatePrevious(double dt, int cframe, double *m_force, int target)
{
// 	bContactForce = false;
// 	time = dt*cframe;
// 	forceCalculator::calculateForceVector(previous, masses);
// // 	for(int i(0); i < 6; i++){
// // 		if(m_force[i])
// // 		{
// // 			bool pause = true;
// // 		}
// // 	}
// // 	if(m_force){
// // 		if(dot(make_vector3d(m_force[0], m_force[1], m_force[2])))
// // 		{
// // 			bContactForce = true;
// // 		}
// // 		//vector4d n = transpose(G(masses->getParameterv(target+1)), make_vector3d(&m_force[3]));
// // 		if(m_force){
// // 			previous.plus(target*7, m_force, 3);
// // 			//previous.plus(target*7+3, POINTER4(n), 4);
// // 		}
// // 	}
// 	constraintCalculator::sparseConstraintJacobian(mDim, cjaco, masses, kconsts);
// 	for(int i(0); i < cjaco.nnz(); i++){
// 		previous(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mDim]; 
// 	}
// 	previous *= alpha / (1 + alpha);
// 	for(int i(0); i < mDim; i++){
// 		ipp(i) = masses->pos(i+7) + masses->vel(i+7)*dt + rhs(i)*dt2accp;
// 		ipv(i) = masses->vel(i+7) + rhs(i)*dt2accv;
// 	}
// 
// 	predictor.apply(cframe);
// 
// 	for(int i(0); i < mDim; i++){
// 		masses->pos(i+7) = ipp(i) + rhs(i)*dt2acc;
// 		masses->vel(i+7) = ipv(i) + rhs(i)*dt*gamma;
// 	}
// 
// 	masses->updateTransformationMatrix();
}

double Mbdsimulation::calculateCorrector(double dt, int cframe, double* m_force/* =NULL */, int target/* =0 */)
{
// 	forceCalculator::calculateForceVector(ee, masses);
// 
// // 	if(m_force && bContactForce){
// // 		//vector4d n = transpose(G(masses->getParameterv(target+1)), make_vector3d(&m_force[3]));
// // 		ee.plus(target*7, m_force, 3);
// // 		//ee.plus(target*7+3, POINTER4(n), 4);
// // 	}
// 	constraintCalculator::sparseConstraintJacobian(mDim, cjaco, masses, kconsts);
// 	for(int i(0); i < cjaco.nnz(); i++){
// 		ee(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mDim]; 
// 	}
// 	massCalculator::calculateMassMatrix(lhs, masses, divalpha);
// 	for(int i=1, k=0; i < masses->size(); i++, k+=7){
// 		ee(k+0) += -lhs(k+0,k+0) * rhs(k+0);
// 		ee(k+1) += -lhs(k+1,k+1) * rhs(k+1);
// 		ee(k+2) += -lhs(k+2,k+2) * rhs(k+2);
// 		for(int j=3; j < 7; j++){
// 			ee(k+j) += -(lhs(k+j,k+3)*rhs(k+3) + lhs(k+j,k+4)*rhs(k+4) + lhs(k+j,k+5)*rhs(k+5) + lhs(k+j,k+6)*rhs(k+6));
// 		} 
// 	}
// 	massCalculator::calculateSystemJacobian(lhs, masses, divalpha*beta*dt*dt);
// 	constraintCalculator::calculateSystemJacobian(lhs, lagMul, masses, kconsts, beta*dt*dt);
// 	for(int i(0); i < cjaco.nnz(); i++){
// 		lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
// 	}
// 	//std::cout << lhs;
// 	for(int i(0); i < mDim; i++) ee(i) -= previous(i);
// 	constraintCalculator::constraintEquation(ee.get_ptr() + mDim, masses, kconsts, dconsts, tDim - mDim, cframe*dt, divbeta);
// 	e_norm = ee.norm();
// 	dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof,	permutation, ee.get_ptr(), ptDof, &lapack_info);
// 	rhs+=ee;
// 	for(int i(0); i < mDim; i++){
// 		masses->pos(i+7) = ipp(i) + rhs(i)*dt2acc;
// 		masses->vel(i+7) = ipv(i) + rhs(i)*dt*gamma;
// 	}
// 	masses->updateTransformationMatrix();
 	return e_norm;
}