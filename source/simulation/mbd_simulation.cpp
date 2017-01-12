#include "mbd_simulation.h"
#include "kinematicConstraint.h"
#include "modeler.h"
#include "object.h"
#include "mass.h"
#include "cylinder.h"
#include <QDebug>
#include <QTime>
#include <QTextStream>
#include <QFile>
#include <cusolverDn.h>


// #ifndef F2C_INCLUDE
// #include <lapack/f2c.h>
// #endif
// #ifndef __CLAPACK_H
// #include <lapack/clapack.h>
// #endif

mbd_simulation::mbd_simulation()
	: simulation()
	, outCount(0)
	, permutation(NULL)
{

}

mbd_simulation::mbd_simulation(modeler *_md)
	: simulation(_md)
	, outCount(0)
	, permutation(NULL)
{
	ground.setID(-1);
	ground.makeTransformationMatrix();
}

mbd_simulation::~mbd_simulation()
{
	//qf_out.close();
	if (qf_out.isOpen())
		qf_out.close();
	if (permutation) delete[] permutation; permutation = NULL;
}

bool mbd_simulation::initialize(bool isCpu)
{
	outCount = 0;
	_isWait = false;
	_isWaiting = false;
	_abort = false;
	_interrupt = false;
	nstep = static_cast<unsigned int>((et / dt) + 1);
	int sdim = 0;
	mbd_simulation::nnz = 0;
	lapack_one = 1;
	lapack_info = 0;
	alpha = -0.3;
	beta = (1 - alpha) * (1 - alpha) / 4;
	gamma = 0.5 - alpha;
	eps = 1E-6;
	unsigned int nm = md->numMass();

	if (nm)
	{
		mdim = nm * 7;
	}

	mass *ground = md->makeMass("ground");
	int i = 0;
	//mass *jbody = md->objects().find("cylinder").value()->pointMass();//md->objects().begin().value()->pointMass();
	//jbody++;
	//md->makeKinematicConstraint("rev0", REVOLUTE, VEC3D(0, 0, 0), NULL, VEC3D(1, 0, 0), VEC3D(0, 1, 0), jbody, VEC3D(1, 0, 0), VEC3D(0, 1, 0));
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++, i++){
		it.value()->setID(i);
		object* o = it.key();
		if (o == NULL) continue;
		switch (o->objectType()){
		case CYLINDER:{
			QMap<QString, cylinder>::iterator cy = md->objCylinders().find(o->objectName());
			it.value()->setEP(cy.value().orientation().To<double>());
			}
			break;
		}
	}
	for (kConstIterator it = md->kinConstraint().begin(); it != md->kinConstraint().end(); it++){
		kinematicConstraint *kconst = it.value();
		kconst->setStartRow(sdim);
		kconst->setFirstColumn(kconst->iMass() ? (kconst->iMass()->ID()-1) * 7 : -1);
		kconst->setSecondColumn(kconst->jMass() ? (kconst->jMass()->ID()-1) * 7 : -1);
		switch (kconst->constType())
		{
		case REVOLUTE:			sdim += 5; break;
		default:
			break;
		}
		nnz += kconst->maxNNZ();
	}
	for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		object* o = it.key();//mass* m = it.value();
		if (o){
			o->updateFromMass();
		}
	}
	dof = mdim - sdim;
	sdim += md->pointMasses().size() - 1;
	tdim = mdim + sdim;

	ptDof = (MKL_INT*)&tdim;
	permutation = new MKL_INT[tdim];
	lhs.alloc(tdim, tdim);
	rhs.alloc(tdim);
	pre.alloc(mdim);
	ipp.alloc(mdim);
	ipv.alloc(tdim);
	ee.alloc(tdim);
	cEQ.alloc(tdim - mdim);
	dt2accp = simulation::dt*simulation::dt*(1 - 2 * beta)*0.5;
	dt2accv = simulation::dt*(1 - gamma);
	dt2acc = simulation::dt*simulation::dt*beta;
	divalpha = 1 / (1 + alpha);
	divbeta = -1 / (beta*simulation::dt*simulation::dt);
	cjaco.alloc(nnz + (nm) * 4, tdim - mdim, mdim);
	FULL_LEOM();
// 	for (kConstIterator it = md->kinConstraint().begin(); md->kinConstraint().end(); it++){
// 
// 	}
	lhs.display();
// 	cusolverDnHandle_t cuhandle = NULL;
// 	int status = 0;
// 	cusolverDnCreate(&cuhandle);
// 	cusolverDnDgetrs(cuhandle, CUBLAS_OP_N, tdim, tdim, lhs.getDataPointer(), tdim, permutation, rhs.get_ptr(), tdim, &status);
// 	cusolverDnDestroy(cuhandle);
	dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof, permutation, rhs.get_ptr(), ptDof, &lapack_info);
	i = 0;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++)
	{
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		int idx = i * 7;
		m->setAcceleration(VEC3D(rhs(idx + 0), rhs(idx * 7 + 1), rhs(idx * 7 + 2)));
		m->setEA(EPD(rhs(idx + 3), rhs(idx + 4), rhs(idx + 5), rhs(idx + 6)));
		i++;
	}

	lagMul = rhs.get_ptr() + mdim;
	qf_out.setFileName(md->modelPath() + "/" + md->modelName() + ".mrf");
	qf_out.open(QIODevice::WriteOnly);
	qf_out.write((char*)&nm, sizeof(unsigned int));
// 	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
// 		object* obj = it.key();
// 		qf_out.write(obj->objectName().toStdString().c_str(), sizeof(char)*obj->objectName().length());
// 	}
	//tp.setFileName("C:/C++/one_pendulum.txt");
	//tp.open(QIODevice::WriteOnly);

	return true;
}

void mbd_simulation::calcMassMatrix(double mul /* = 1.0 */)
{
	int cnt = 0;
	lhs.zeros();
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		lhs(cnt, cnt) = lhs(cnt + 1, cnt + 1) = lhs(cnt + 2, cnt + 2) = mul * m->getMass();
		cnt += 3;
		MAT44D LTJL = 4.0f*mul*transpose(m->getEP().G(), m->getInertia() * m->getEP().G());
		for (int j(0); j < 4; j++){
			for (int k(0); k < 4; k++){
				if (LTJL(j, k))
					lhs(cnt + j, cnt + k) = LTJL(j, k);
			}
		}
		cnt += 4;
	}
}

VEC4D mbd_simulation::calcInertiaForce(EPD& ev, MAT33D& J, EPD& ep)
{
	double GvP0 = -ev.e1*ep.e0 + ev.e0*ep.e1 + ev.e3*ep.e2 - ev.e2*ep.e3;
	double GvP1 = -ev.e2*ep.e0 - ev.e3*ep.e1 + ev.e0*ep.e2 + ev.e1*ep.e3;
	double GvP2 = -ev.e3*ep.e0 + ev.e2*ep.e1 - ev.e1*ep.e2 + ev.e0*ep.e3;
	return VEC4D(
		8*(-ev.e1*J.a00*GvP0 - ev.e2*J.a11*GvP1 - ev.e3*J.a22*GvP2),
		8*(ev.e0*J.a00*GvP0 - ev.e3*J.a11*GvP1 + ev.e2*J.a22*GvP2),
		8*(ev.e3*J.a00*GvP0 + ev.e0*J.a11*GvP1 - ev.e1*J.a22*GvP2),
		8*(-ev.e2*J.a00*GvP0 + ev.e1*J.a11*GvP1 + ev.e0*J.a22*GvP2));
}

VEC4D mbd_simulation::calcMoment(EPD& ep, VEC3D& m)
{
	VEC4D out;
	out = transpose(2 * m, ep.L());
	///*VEC4D*/ out = transpose(2 * m, ep.G());
	return out;
}

void mbd_simulation::calcForceVector(VECD* vec)
{
	VECD *out = vec ? vec : &rhs;
	out->zeros();
	int nmass = md->pointMasses().size();
	int i = 1;
	int cnt = 0;
	VEC3D nf;
	VEC4D mm, im, rf;
	VEC3D g = md->gravity().To<double>();
	VEC3D cm;
// 	VEC3D add_moment = VEC3D(0, 0, 0);
//  	if ((ct >= 0.1f && ct < 0.25f)/* && (simulation::ct < 0.35)*/)
//  		add_moment = (ct - 0.1f) * VEC3D(0, 0, 1.5);
//  	else if (ct >= 0.20f && ct < 0.3f)
//  		add_moment = VEC3D(0, 0, 0);
// 	else if (ct >= 0.3f)
// 		add_moment = VEC3D(0, 0, -4);
// // 		add_moment = VEC3D(0, 0, 0.1);
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		nf = m->getMass() * g + m->getCollisionForce() + m->getExternalForce();
	///	VEC3D _mm = (VEC3D(0.4, 0.1, 0.0) - m->getPosition()).cross(VEC3D(0.0, -10, 0.0));
// 		if (m->getExternalMoment().length())
// 			cm = m->getCollisionForce()
		mm = calcMoment(m->getEP(), m->getCollisionMoment() + m->getExternalMoment());
		rf = calcInertiaForce(m->getEV(), m->getInertia(), m->getEP());
		//rf.x += mm.x; rf.y += mm.y; rf.z += mm.z; rf.w += mm.w;
		rf += mm;
		out->insert(cnt, POINTER3(nf), POINTER4(rf), 3, 4);
		cnt += 7;
	}

}

void mbd_simulation::sparseConstraintJacobian()
{
	int sr, ic, jc, i = 0;
	mass *ib, *jb;
	VEC3D dij;
	EPD ep;
	cjaco.zeroCount();
	sr = tdim - md->pointMasses().size() + 1;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		EPD ep2 = 2 * m->getEP();
		cjaco.extraction(sr++, i * 7 + 3, ep2.Pointer(), 4);
		i++;
	}
	for (kConstIterator it = md->kinConstraint().begin(); it != md->kinConstraint().end(); it++){
		kinematicConstraint* kconst = it.value();
		sr = kconst->startRow() + (md->pointMasses().size() - 1) * 7;
		ib = kconst->iMass(); ic = kconst->iColumn();
		jb = kconst->jMass(); jc = kconst->jColumn();
		switch (kconst->constType())
		{
		case REVOLUTE:
			if (ib)
			{
				if (!jb) jb = &ground;
				for (unsigned i(0); i < 3; i++) cjaco(sr + i, ic + i) = -1;
				ep = ib->getEP();// m->getParameterv(ib);
				cjaco.extraction(sr + 0, ic + 3, POINTER(B(ep, -kconst->sp_i())), MAT3X4);
				cjaco.extraction(sr + 3, ic + 3, POINTER(transpose(jb->toGlobal(kconst->h_j()), B(ep, kconst->g_i()))), VEC4);
				cjaco.extraction(sr + 4, ic + 3, POINTER(transpose(jb->toGlobal(kconst->h_j()), B(ep, kconst->f_i()))), VEC4);
			}
			if (jb)
			{
				if (!ib) ib = &ground;
				for (unsigned i(0); i < 3; i++) cjaco(sr + i, jc + i) = 1;
				ep = jb->getEP();
				cjaco.extraction(sr + 0, jc + 3, POINTER(B(ep, kconst->sp_j())), MAT3X4);
				cjaco.extraction(sr + 3, jc + 3, POINTER(transpose(ib->toGlobal(kconst->g_i()), B(ep, kconst->h_j()))), VEC4);
				cjaco.extraction(sr + 4, jc + 3, POINTER(transpose(ib->toGlobal(kconst->f_i()), B(ep, kconst->h_j()))), VEC4);
			}
			//std::cout << *sjc << std::endl;
			break;
		}
	}
}

void mbd_simulation::FULL_LEOM()
{
	calcMassMatrix();
	calcForceVector();
	sparseConstraintJacobian();
	for (int i(0); i < cjaco.nnz(); i++){
		lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
	}
	//lhs.display();
}

bool mbd_simulation::saveResult(float ct, unsigned int p)
{
	unsigned int cnt = 0;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++)
	{
		//object* obj = it.key();
		mass *m = it.value();
		if (m->ID() == 0)
			continue;
		//unsigned int id = obj->ID();
		qf_out.write((char*)&cnt, sizeof(unsigned int));
		qf_out.write((char*)&ct, sizeof(float));
		qf_out.write((char*)&(m->getPosition()), sizeof(VEC3D));
		qf_out.write((char*)&(m->getEP()), sizeof(EPD));
		qf_out.write((char*)&(m->getVelocity()), sizeof(VEC3D));
		qf_out.write((char*)&(m->getEV()), sizeof(EPD));
		qf_out.write((char*)&(m->getAcceleration()), sizeof(VEC3D));
		qf_out.write((char*)&(m->getEA()), sizeof(EPD));
		cnt++;
	}
	outCount++;
 	return true;
}

void mbd_simulation::prediction(unsigned int cs)
{
	unsigned int i = 0;
	calcForceVector(&pre);
	sparseConstraintJacobian();
	for (i = 0; i < cjaco.nnz(); i++){
		pre(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mdim];
	}
	pre *= alpha / (1 + alpha);
	i = 0;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		unsigned int idx = i * 7;
		for (int j = 0; j < 3; j++){
			ipp(idx + j) = m->getPosition()(j)+m->getVelocity()(j)* simulation::dt + rhs(idx + j) * dt2accp;
			ipv(idx + j) = m->getVelocity()(j)+rhs(idx + j) * dt2accv;
		}
		idx += 3;
		for (int j = 0; j < 4; j++){
			ipp(idx + j) = m->getEP()(j)+m->getEV()(j)* simulation::dt + rhs(idx + j) * dt2accp;
			ipv(idx + j) = m->getEV()(j)+rhs(idx + j) * dt2accv;
		}
		i++;
	}

	i = 0;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		VEC3D p, v;
		EPD ep, ev;
		unsigned int idx = i * 7;
		for (int j = 0; j < 3; j++){
			p(j) = ipp(idx + j) + rhs(idx + j) * dt2acc;
			v(j) = ipv(idx + j) + rhs(idx + j) * dt * gamma;
		}
		idx += 3;
		for (int j = 0; j < 4; j++){
			ep(j) = ipp(idx + j) + rhs(idx + j) * dt2acc;
			ev(j) = ipv(idx + j) + rhs(idx + j) * dt * gamma;
		}
		//std::cout << ipv(1) << std::endl;
		m->setPosition(p);
		m->setVelocity(v);
		m->setEP(ep);
		m->setEV(ev);
		ep.e0 = 1.0 - (ep.e1 * ep.e1 + ep.e2 * ep.e2 + ep.e3 * ep.e3);
		m->makeTransformationMatrix();
		i++;
	}
}

void mbd_simulation::calcMassSystemJacobian(double mul)
{
	int sr = 0;
	EPD e;
	EPD edd;
	MAT33D inertia;
	MAT44D data;
	unsigned int i = 0;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		sr = i * 7 + 3;
		e = m->getEP();
		edd = m->getEA();
		inertia = m->getInertia();
		data = mul * (-transpose(e.G(), inertia * edd.G()) + opMiner(inertia * (e.G() * edd)));
		lhs.plus(sr, sr, POINTER(data), MAT4x4);
		i++;
	}
}

MAT44D mbd_simulation::D(VEC3D& a, VEC3D& lag)
{
	double one = 2 * (a.x*lag.x + a.y*lag.y + a.z*lag.z);
	VEC3D upper = transpose(a, tilde(lag));
	VEC3D left = a.cross(lag);
	MAT33D mid(
		-a.z*lag.z - a.y*lag.y - lag.z*a.z - lag.y*a.y + one,								 a.y*lag.y + lag.y*a.x,								    a.z*lag.x + lag.z*a.x,
									   a.x*lag.y + lag.x*a.y, -a.z*lag.z - a.x*lag.x - lag.z*a.z - lag.x*a.x + one,									a.z*lag.y + lag.z*a.y,
									   a.x*lag.z + lag.x*a.z,								 a.y*lag.z + lag.y*a.z,	-a.y*lag.y - a.x*lag.x - lag.y*a.y - lag.x*a.x + one);
	return MAT44D(
		2 * one, upper.x, upper.y, upper.z,
		left.x, mid.a00, mid.a01, mid.a02,
		left.y, mid.a10, mid.a11, mid.a12,
		left.z, mid.a20, mid.a21, mid.a22);
}

void mbd_simulation::calcConstraintSystemJacobian(double mul)
{
	int sr = 0;
	int ic = 0;
	int jc = 0;
	mass* ib = NULL;
	mass* jb = NULL;
	MAT44D Dv;
	MAT34D Bv;
	MATD m_lhs(lhs.rows(), lhs.cols()); m_lhs.zeros();
	for (kConstIterator it = md->kinConstraint().begin(); it != md->kinConstraint().end(); it++){
		kinematicConstraint *kconst = it.value();
		ib = kconst->iMass();
		jb = kconst->jMass();
		ic = kconst->iColumn();
		jc = kconst->jColumn();
		switch (kconst->constType())
		{
		case REVOLUTE:
			if (ib){
				if (!jb) jb = &ground;
				Dv = -D(kconst->sp_i(), VEC3D(lagMul[sr + 0], lagMul[sr + 1], lagMul[sr + 2]));
				m_lhs.plus(ic + 3, ic + 3, POINTER(Dv), MAT4x4);
				Dv = lagMul[sr + 3] * D(kconst->g_i(), jb->toGlobal(kconst->h_j())) + lagMul[sr + 4] * D(kconst->f_i(), jb->toGlobal(kconst->h_j()));
				m_lhs.plus(ic + 3, ic + 3, POINTER(Dv), MAT4x4);
				if (jb != &ground){
					Dv = lagMul[sr + 3] * transpose(B(jb->getEP(), kconst->h_j()), B(ib->getEP(), kconst->g_i())) + lagMul[sr + 4] * transpose(B(jb->getEP(), kconst->h_j()), B(ib->getEP(), kconst->f_i()));
					m_lhs.plus(jc + 3, ic + 3, POINTER(Dv), MAT4x4);
				}
			}
			if (jb){
				if (!ib) ib = &ground;
				Dv = D(kconst->sp_j(), VEC3D(lagMul[sr + 0], lagMul[sr + 1], lagMul[sr + 2]));
				m_lhs.plus(jc + 3, jc + 3, POINTER(Dv), MAT4x4);
				Dv = lagMul[sr + 3] * D(kconst->h_j(), ib->toGlobal(kconst->g_i())) + lagMul[sr + 4] * D(kconst->h_j(), ib->toGlobal(kconst->f_i()));
				m_lhs.plus(jc + 3, jc + 3, POINTER(Dv), MAT4x4);
				if (ib != &ground){
					Dv = lagMul[sr + 3] * transpose(B(ib->getEP(), kconst->g_i()), B(jb->getEP(), kconst->h_j())) + lagMul[sr + 4] * transpose(B(ib->getEP(), kconst->f_i()), B(jb->getEP(), kconst->h_j()));
					m_lhs.plus(ic + 3, jc + 3, POINTER(Dv), MAT4x4);
				}
			}
			sr += 5;
			break;
		}
	}
	for (unsigned int i = 0; i < m_lhs.rows(); i++){
		for (unsigned int j = 0; j < m_lhs.cols(); j++){
			lhs(i, j) += mul * m_lhs(i, j);
		}
	}
}

void mbd_simulation::constraintEquation()
{
	double* rhs = ee.get_ptr() + mdim;
	VEC3D v3;
	mass *ib = NULL;
	mass *jb = NULL;

	for (kConstIterator it = md->kinConstraint().begin(); it != md->kinConstraint().end(); it++){
		kinematicConstraint *kconst = it.value();
		ib = kconst->iMass(); if (!ib) ib = &ground;
		jb = kconst->jMass(); if (!jb) jb = &ground;
		switch (kconst->constType()){
		case REVOLUTE:
			v3 = jb->getPosition() + jb->toGlobal(kconst->sp_j()) - ib->getPosition() - ib->toGlobal(kconst->sp_i());
			rhs[kconst->startRow() + 0] = divbeta * v3.x;
			rhs[kconst->startRow() + 1] = divbeta * v3.y;
			rhs[kconst->startRow() + 2] = divbeta * v3.z;
			v3 = jb->toGlobal(kconst->h_j());
			rhs[kconst->startRow() + 3] = divbeta * v3.dot(ib->toGlobal(kconst->g_i()));
			rhs[kconst->startRow() + 4] = divbeta * v3.dot(ib->toGlobal(kconst->f_i()));
			break;
		}
	}
	int mr = tdim - (md->pointMasses().size()-1) - mdim;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		rhs[mr++] = divbeta * (m->getEP().dot() - 1.0);
	}
}

double mbd_simulation::correction(unsigned int cs)
{
	double e_norm = 1;
	while (1){
		calcForceVector(&ee);
		sparseConstraintJacobian();
		for (int i = 0; i < cjaco.nnz(); i++){
			ee(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mdim];
		}

		calcMassMatrix(divalpha);
		for (unsigned int i = 0, k = 0; i < md->pointMasses().size(); i++, k += 7){
			ee(k + 0) += -lhs(k + 0, k + 0) * rhs(k + 0);
			ee(k + 1) += -lhs(k + 1, k + 1) * rhs(k + 1);
			ee(k + 2) += -lhs(k + 2, k + 2) * rhs(k + 2);
			for (unsigned int j = 3; j < 7; j++){
				ee(k + j) += -(lhs(k + j, k + 3) * rhs(k + 3) + lhs(k + j, k + 4)*rhs(k + 4) + lhs(k + j, k + 5)*rhs(k + 5) + lhs(k + j, k + 6)*rhs(k + 6));
			}
		}
		calcMassSystemJacobian(divalpha * beta * dt * dt);
		calcConstraintSystemJacobian(beta * dt * dt);
		for (int i(0); i < cjaco.nnz(); i++){
			lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
		}
		for (int i(0); i < mdim; i++) ee(i) -= pre(i);
		constraintEquation();
		e_norm = ee.norm();
// 		cusolverDnHandle_t cuhandle = NULL;
// 		int status = 0;
// 		cusolverDnCreate(&cuhandle);
// 		cusolverDnDgetrs(cuhandle, CUBLAS_OP_N, tdim, tdim, lhs.getDataPointer(), tdim, permutation, rhs.get_ptr(), tdim, &status);
// 		cusolverDnDestroy(cuhandle);
		dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof, permutation, ee.get_ptr(), ptDof, &lapack_info);
		rhs += ee;
		int idx = 0;
		for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
			mass* m = it.value();
			if (m->ID() == 0)
				continue;
			VEC3D p, v;
			EPD ep, ev;
			for (int i = 0; i < 3; i++){
				p(i) = ipp(idx + i) + dt2acc * rhs(idx + i);
				v(i) = ipv(idx + i) + dt * gamma * rhs(idx + i);
			}
			m->setAcceleration(VEC3D(rhs(idx + 0), rhs(idx + 1), rhs(idx + 2)));
			idx += 3;
			for (int i = 0; i < 4; i++){
				ep(i) = ipp(idx + i) + dt2acc * rhs(idx + i);
				ev(i) = ipv(idx + i) + dt * gamma * rhs(idx + i);
			}
			m->setEA(EPD(rhs(idx + 0), rhs(idx + 1), rhs(idx + 2), rhs(idx + 3)));
			m->setPosition(p);
			m->setVelocity(v);
			
			m->setEP(ep);
			m->setEV(ev);
			idx += 4;
			m->makeTransformationMatrix();
		}
		if (e_norm <= 1e-5)
		{
// 			for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++)
// 			{
// 				it.value()->setExternalForce(0);
// 				it.value()->setExternalMoment(0);
// 			}
			break;
		}
	}
	return e_norm;
}

double mbd_simulation::oneStepCorrection()
{
	double e_norm = 1;
	calcForceVector(&ee);
	sparseConstraintJacobian();
	for (int i = 0; i < cjaco.nnz(); i++){
		ee(cjaco.cidx[i]) -= cjaco.value[i] * lagMul[cjaco.ridx[i] - mdim];
	}

	calcMassMatrix(divalpha);
	for (unsigned int i = 0, k = 0; i < md->pointMasses().size()-1; i++, k += 7){
		ee(k + 0) += -lhs(k + 0, k + 0) * rhs(k + 0);
		ee(k + 1) += -lhs(k + 1, k + 1) * rhs(k + 1);
		ee(k + 2) += -lhs(k + 2, k + 2) * rhs(k + 2);
		for (unsigned int j = 3; j < 7; j++){
			ee(k + j) += -(lhs(k + j, k + 3) * rhs(k + 3) + lhs(k + j, k + 4)*rhs(k + 4) + lhs(k + j, k + 5)*rhs(k + 5) + lhs(k + j, k + 6)*rhs(k + 6));
		}
	}
	//calcMassSystemJacobian(divalpha * beta * dt * dt);
	//calcConstraintSystemJacobian(beta * dt * dt);
	for (int i(0); i < cjaco.nnz(); i++){
		lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
	}
	for (int i(0); i < mdim; i++) ee(i) -= pre(i);
	constraintEquation();
	e_norm = ee.norm();
// 	cusolverDnHandle_t cuhandle = NULL;
// 	int status = 0;
// 	cusolverDnCreate(&cuhandle);
// 	cusolverDnDgetrs(cuhandle, CUBLAS_OP_N, tdim, tdim, lhs.getDataPointer(), tdim, permutation, rhs.get_ptr(), tdim, &status);
// 	cusolverDnDestroy(cuhandle);
 	dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof, permutation, ee.get_ptr(), ptDof, &lapack_info);
 	rhs += ee;
	int idx = 0;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		VEC3D p, v;
		EPD ep, ev;
		for (int i = 0; i < 3; i++){
			p(i) = ipp(idx + i) + dt2acc * rhs(idx + i);
			v(i) = ipv(idx + i) + dt * gamma * rhs(idx + i);
		}
		m->setAcceleration(VEC3D(rhs(idx + 0), rhs(idx + 1), rhs(idx + 2)));
		idx += 3;
		for (int i = 0; i < 4; i++){
			ep(i) = ipp(idx + i) + dt2acc * rhs(idx + i);
			ev(i) = ipv(idx + i) + dt * gamma * rhs(idx + i);
		}
		m->setEA(EPD(rhs(idx + 0), rhs(idx + 1), rhs(idx + 2), rhs(idx + 3)));
		m->setPosition(p);
		m->setVelocity(v);
		m->setEP(ep);
		//m->getEP().
		//m->getEP().normalize();
		m->setEV(ev);
		idx += 4;
		m->makeTransformationMatrix();
	}
	return e_norm;
}

bool mbd_simulation::cpuRun()
{
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

	while (cStep < nstep)
	{
		if (_isWait)
			continue;
		if (_abort){
			emit finished();
			return false;
		}
		ct = dt * cStep;
		md->runExpression(ct, dt);
		prediction(cStep);
		correction(cStep);
		if (!((cStep) % step)){
			part++;
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

bool mbd_simulation::gpuRun()
{
	return true;
}