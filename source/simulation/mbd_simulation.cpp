#include "mbd_simulation.h"
#include "kinematicConstraint.h"
#include "drivingConstraint.h"
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
	EPD _ep;
	_ep.setFromEuler(0, 0.5 * M_PI, 0);
	double n = _ep.dot();
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++, i++){
		object* o = it.key();
		if (o == NULL) continue;
		it.value()->setID(i);
		switch (o->objectType()){
		case CYLINDER:{
			cylinder* cy = md->getChildObject<cylinder*>(o->objectName());
			it.value()->setEP(cy->orientation());
			it.value()->makeTransformationMatrix();
			}
			break;
		}
	}
	mass* shaft;
	mass* guide; 
	mass* wheel;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++, i++){
		mass* m = it.value();
		if (m->name() == "shaft")
			shaft = m;
		else if (m->name() == "guide")
			guide = m;
		else if (m->name() == "wheel")
			wheel = m;
	}

	mass* m = (++md->pointMasses().begin()).value();
	kinematicConstraint* rev_ws = md->makeKinematicConstraint("rev_wheel_ground", REVOLUTE, wheel, VEC3D(0, 0, 0), wheel->toLocal(VEC3D(1.0, 0, 0)), wheel->toLocal(VEC3D(0, 1.0, 0)),
		ground, VEC3D(0, -0.012, 0), VEC3D(-1.0, 0, 0), VEC3D(0, 1.0, 0));
// 	md->makeKinematicConstraint("tra_shaft_guide", TRANSLATIONAL, VEC3D(0, 0.288, 0), shaft, shaft->toLocal(VEC3D(0, 0, 1.0)), shaft->toLocal(VEC3D(1.0, 0, 0)),
// 								ground, ground->toLocal(VEC3D(1.0, 0, 0)), ground->toLocal(VEC3D(0, 0, -1.0)));
// 	md->makeKinematicConstraint("tra_shaft_guide", TRANSLATIONAL,  shaft, shaft->getPosition(), shaft->toLocal(VEC3D(0, 0, 1.0)), shaft->toLocal(VEC3D(1.0, 0, 0)),
// 								guide, guide->getPosition(), guide->toLocal(VEC3D(1.0, 0, 0)), guide->toLocal(VEC3D(0, 0, -1.0)));
// 	md->makeKinematicConstraint("tra_guide_ground", TRANSLATIONAL, guide, guide->getPosition(), guide->toLocal(VEC3D(0, 1.0, 0)), guide->toLocal(VEC3D(0, 0, -1.0)),
// 								ground, VEC3D(-1, 0.225, 0), ground->toLocal(VEC3D(0, 0, 1.0)), ground->toLocal(VEC3D(0, 1.0, 0)));

	md->makeDrivingConstraint("Wheel_driving", rev_ws, DRIVING_VELOCITY, 1);

	for (kConstIterator it = md->kinConstraint().begin(); it != md->kinConstraint().end(); it++){
		kinematicConstraint *kconst = it.value();
		kconst->setStartRow(sdim);
		kconst->setFirstColumn(kconst->iMass() ? (kconst->iMass()->ID()-1) * 7 : -1);
		kconst->setSecondColumn(kconst->jMass() ? (kconst->jMass()->ID()-1) * 7 : -1);
		switch (kconst->constType())
		{
		case REVOLUTE:			sdim += 5; break;
		case TRANSLATIONAL:		sdim += 5; break;
		default:
			break;
		}
		nnz += kconst->maxNNZ();
	}
	for (dConstIterator it = md->drivingConstraints().begin(); it != md->drivingConstraints().end(); it++){
		drivingConstraint* dconst = it.value();
		dconst->setStartRow(sdim);
		sdim++;
		nnz += dconst->maxNNZ();
		//dconst->setStartColumn()
	}
	//sdim += md->drivingConstraints().size();
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
	ipv.alloc(mdim);
	ee.alloc(tdim);
	cEQ.alloc(tdim - mdim);
	dt2accp = simulation::dt*simulation::dt*(1 - 2 * beta)*0.5;
	dt2accv = simulation::dt*(1 - gamma);
	dt2acc = simulation::dt*simulation::dt*beta;
	divalpha = 1 / (1 + alpha);
	divbeta = -1 / (beta*simulation::dt*simulation::dt);
	cjaco.alloc(nnz + (nm) * 4, tdim - mdim, mdim);
// 	EPD ev0 = wheel->getEP().w2ev(VEC3D(0, 0, 0.2));
// 	wheel->setEV(ev0);
	FULL_LEOM();
	//lhs.display();
	dgesv_(ptDof, &lapack_one, lhs.getDataPointer(), ptDof, permutation, rhs.get_ptr(), ptDof, &lapack_info);
	i = 0;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++)
	{
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		int idx = i * 7;
		m->setAcceleration(VEC3D(rhs(idx + 0), rhs(idx + 1), rhs(idx + 2)));
		m->setEA(EPD(rhs(idx + 3), rhs(idx + 4), rhs(idx + 5), rhs(idx + 6)));
		i++;
	//	m->setForceAndMomentMemory(nstep);
	}

	lagMul = rhs.get_ptr() + mdim;
	qf_out.setFileName(md->modelPath() + "/" + md->modelName() + ".mrf");
	qf_out.open(QIODevice::WriteOnly);
	qf_out.write((char*)&nm, sizeof(unsigned int));
	unsigned int nr_out = static_cast<unsigned int>(nstep / step) + 1;
	qf_out.write((char*)&nr_out, sizeof(unsigned int));

	//unsigned int nr_out = static_cast<unsigned int>(nstep / step) + 1;
	for (QMap<object*, mass*>::iterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		if (it.value()->ID() == 0)
			continue;
		if (it.value()->getBaseGeometryType() == POLYGON)
			it.value()->getPolyGeometryObject()->setResultData(nr_out);
		else
			it.value()->getGeometryObject()->setResultData(nr_out);
	}
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
	VEC3D g = md->gravity();
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
		//if (m->getCollisionForce())
		
		nf = m->getMass() * g + m->getCollisionForce() + m->getExternalForce();
		
				
	///	VEC3D _mm = (VEC3D(0.4, 0.1, 0.0) - m->getPosition()).cross(VEC3D(0.0, -10, 0.0));
// 		if (m->getExternalMoment().length())
// 			cm = m->getCollisionForce()
// 		if (m->name() == "wheel" && ct != 0)
// 		{
// 			m->setCollisionMoment(VEC3D(0, 0, 100));
// 		}
		mm = calcMoment(m->getEP(), m->getCollisionMoment() + m->getExternalMoment()/* + VEC3D(0, 0, 1)*/);
// 		if (m->name() == "wheel")
// 		{
// 			mm.z += 1.0;
// 		}
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
	//VEC3D dij;
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
			if (ib->ID())
			{
				//if (!jb->ID()) jb = &ground;
				for (unsigned i(0); i < 3; i++) cjaco(sr + i, ic + i) = -1;
				ep = ib->getEP();// m->getParameterv(ib);
				cjaco.extraction(sr + 0, ic + 3, POINTER(B(ep, -kconst->sp_i())), MAT3X4);
				cjaco.extraction(sr + 3, ic + 3, POINTER(transpose(jb->toGlobal(kconst->h_j()), B(ep, kconst->g_i()))), VEC4);
				cjaco.extraction(sr + 4, ic + 3, POINTER(transpose(jb->toGlobal(kconst->h_j()), B(ep, kconst->f_i()))), VEC4);
			}
			if (jb->ID())
			{
				//if (!ib->ID()) ib = &ground;
				for (unsigned i(0); i < 3; i++) cjaco(sr + i, jc + i) = 1;
				ep = jb->getEP();
				cjaco.extraction(sr + 0, jc + 3, POINTER(B(ep, kconst->sp_j())), MAT3X4);
				cjaco.extraction(sr + 3, jc + 3, POINTER(transpose(ib->toGlobal(kconst->g_i()), B(ep, kconst->h_j()))), VEC4);
				cjaco.extraction(sr + 4, jc + 3, POINTER(transpose(ib->toGlobal(kconst->f_i()), B(ep, kconst->h_j()))), VEC4);
			}
			//std::cout << *sjc << std::endl;
			break;
		case TRANSLATIONAL:
			//if (!ib) ib = &ground;
			//else if (!jb) jb = &ground;
			VEC3D dij;
// 			if (jb->ID()) dij = jb->getPosition() + jb->toGlobal(kconst->sp_j());
// 			if (ib->ID()) dij -= ib->getPosition() + ib->toGlobal(kconst->sp_i());
			dij = ( jb->getPosition() + jb->toGlobal(kconst->sp_j()) ) - ( ib->getPosition() + ib->toGlobal(kconst->sp_i()) );
			if (ib->ID())
			{
				//if(!(jb->ID())) jb = &ground;
				cjaco.extraction(sr + 0, ic + 3, POINTER(transpose(jb->toGlobal(kconst->h_j()), B(ep, kconst->g_i()))), VEC4);
				cjaco.extraction(sr + 1, ic + 3, POINTER(transpose(jb->toGlobal(kconst->h_j()), B(ep, kconst->f_i()))), VEC4);
				cjaco.extraction(sr + 2, ic + 0, POINTER((-kconst->g_i())), POINTER(transpose(dij + kconst->sp_i(), B(ep, kconst->g_i()))), VEC3_4);
				cjaco.extraction(sr + 3, ic + 0, POINTER((-kconst->f_i())), POINTER(transpose(dij + kconst->sp_i(), B(ep, kconst->f_i()))), VEC3_4);
				cjaco.extraction(sr + 4, ic + 3, POINTER(transpose(jb->toGlobal(kconst->f_j()), B(ep, kconst->f_i()))), VEC4);
			}
			if (jb->ID())
			{
				//if (!(ib->ID())) ib = &ground;
				cjaco.extraction(sr + 0, jc + 3, POINTER(transpose(ib->toGlobal(kconst->g_i()), B(ep, kconst->h_j()))), VEC4);
				cjaco.extraction(sr + 1, jc + 3, POINTER(transpose(ib->toGlobal(kconst->f_i()), B(ep, kconst->h_j()))), VEC4);
				cjaco.extraction(sr + 2, jc + 0, POINTER(kconst->g_i()), POINTER(transpose(kconst->g_i(), B(ep, kconst->sp_j()))), VEC3_4);
				cjaco.extraction(sr + 3, jc + 0, POINTER(kconst->f_i()), POINTER(transpose(kconst->f_i(), B(ep, kconst->sp_j()))), VEC3_4);
				cjaco.extraction(sr + 4, jc + 3, POINTER(transpose(jb->toGlobal(kconst->f_i()), B(ep, kconst->f_j()))), VEC4);
			}
			break;
		}
	}
	for (dConstIterator it = md->drivingConstraints().begin(); it != md->drivingConstraints().end(); it++){
		drivingConstraint* dconst = it.value();
		sr = dconst->startRow() + (md->pointMasses().size() - 1) * 7;
		ic = dconst->startColumn();
		//for (int i = 0; i < 7; i++) 
			//if (dconst->use(i)) 
		cjaco(sr, ic + 3) = 1.0;
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
	lhs.display();
}

bool mbd_simulation::saveResult(double ct, unsigned int p)
{
	unsigned int cnt = 0;

	qf_out.write((char*)&outCount, sizeof(unsigned int));
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++)
	{
		mass *m = it.value();
		if (m->ID() == 0)
			continue;
		unsigned int name_size = m->name().size();
		qf_out.write((char*)&name_size, sizeof(unsigned int));
		qf_out.write((char*)m->name().toStdString().c_str(), sizeof(char) * m->name().size());
		qf_out.write((char*)&cnt, sizeof(unsigned int));
		qf_out.write((char*)&ct, sizeof(double));
		qf_out.write((char*)&(m->getPosition()), sizeof(VEC3D));
		qf_out.write((char*)&(m->getEP()), sizeof(EPD));
		qf_out.write((char*)&(m->getVelocity()), sizeof(VEC3D));
		qf_out.write((char*)&(m->getEV()), sizeof(EPD));
		qf_out.write((char*)&(m->getAcceleration()), sizeof(VEC3D));
		qf_out.write((char*)&(m->getEA()), sizeof(EPD));
		m->getGeometryObject()->insertResultData(outCount, m->getPosition(), m->getEP());
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
// 		double nd = ep.dot();
// 		double n = ep.length();
		m->setEV(ev);
		//ep.e0 = 1.0 - (ep.e1 * ep.e1 + ep.e2 * ep.e2 + ep.e3 * ep.e3);
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
	VEC3D dij;
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
			if (ib->ID()){
				if (!jb->ID()) jb = &ground;
				Dv = -D(kconst->sp_i(), VEC3D(lagMul[sr + 0], lagMul[sr + 1], lagMul[sr + 2]));
				m_lhs.plus(ic + 3, ic + 3, POINTER(Dv), MAT4x4);
				Dv = lagMul[sr + 3] * D(kconst->g_i(), jb->toGlobal(kconst->h_j())) + lagMul[sr + 4] * D(kconst->f_i(), jb->toGlobal(kconst->h_j()));
				m_lhs.plus(ic + 3, ic + 3, POINTER(Dv), MAT4x4);
				if (jb != &ground){
					Dv = lagMul[sr + 3] * transpose(B(jb->getEP(), kconst->h_j()), B(ib->getEP(), kconst->g_i())) + lagMul[sr + 4] * transpose(B(jb->getEP(), kconst->h_j()), B(ib->getEP(), kconst->f_i()));
					m_lhs.plus(jc + 3, ic + 3, POINTER(Dv), MAT4x4);
				}
			}
			if (jb->ID()){
				if (!ib->ID()) ib = &ground;
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
		case TRANSLATIONAL:
			if (ib->ID())
			{
				if (!jb->ID()) jb = &ground;
				Dv = lagMul[sr + 0] * D(kconst->g_i(), jb->toGlobal(kconst->h_j())) + lagMul[sr + 1] * D(kconst->f_i(), jb->toGlobal(kconst->h_j())) + lagMul[sr + 4] * D(kconst->f_i(), jb->toGlobal(kconst->f_j()));
				m_lhs.plus(ic + 3, ic + 3, POINTER(Dv), MAT4x4);
				if (jb != &ground){
					Dv = lagMul[sr + 0] * transpose(B(ib->getEP(), kconst->g_i()), B(jb->getEP(), kconst->h_j())) + lagMul[sr + 1] * transpose(B(ib->getEP(), kconst->f_i()), B(jb->getEP(), kconst->h_j())) + lagMul[sr + 4] * transpose(B(ib->getEP(), kconst->f_i()), B(jb->getEP(), kconst->f_j()));
					m_lhs.plus(ic + 3, jc + 3, POINTER(Dv), MAT4x4);
				}
				dij = (jb->getPosition() + jb->toGlobal(kconst->sp_j())) - (ib->getPosition() + ib->toGlobal(kconst->sp_i()));
				Bv = -lagMul[sr + 2] * B(ib->getEP(), kconst->f_i()) - lagMul[sr + 3] * B(ib->getEP(), kconst->g_i()); 
				m_lhs.plus(ic + 0, ic + 3, POINTER(Bv), MAT3X4);
				m_lhs.plus(ic + 3, ic + 0, POINTER(Bv), MAT4x3);
				Dv = lagMul[sr + 2] * D(kconst->f_i(), dij + ib->toGlobal(kconst->sp_i())) + lagMul[sr + 3] * D(kconst->g_i(), dij + ib->toGlobal(kconst->sp_i()));
				m_lhs.plus(ic + 3, ic + 3, POINTER(Dv), MAT4x4);
				if (jb != &ground)
				{
					Bv = -Bv;
					m_lhs.plus(ic + 3, jc + 0, POINTER(Bv), MAT4x3);
					Dv = lagMul[sr + 2] * transpose(B(ib->getEP(), kconst->f_i()), B(jb->getEP(), kconst->sp_j())) + lagMul[sr + 3] * transpose(B(ib->getEP(), kconst->g_i()), B(jb->getEP(), kconst->sp_j()));
					m_lhs.plus(ic + 3, jc + 3, POINTER(Dv), MAT4x4);
				}
			}
			if (jb)
			{
				if (!ib->ID()) ib = &ground;
				Dv = lagMul[sr + 0] * D(kconst->h_j(), jb->toGlobal(kconst->g_i())) + lagMul[sr + 1] * D(kconst->h_j(), jb->toGlobal(kconst->f_i())) + lagMul[sr + 4] * D(kconst->f_j(), jb->toGlobal(kconst->f_i()));
				m_lhs.plus(jc + 3, jc + 3, POINTER(Dv), MAT4x4);
				if (ib != &ground)
				{
					Dv = lagMul[sr + 0] * transpose(B(jb->getEP(), kconst->h_j()), B(ib->getEP(), kconst->g_i())) + lagMul[sr + 1] * transpose(B(jb->getEP(), kconst->h_j()), B(ib->getEP(), kconst->f_i())) + lagMul[sr + 4] * transpose(B(jb->getEP(), kconst->f_j()), B(ib->getEP(), kconst->f_i()));
					m_lhs.plus(jc + 3, ic + 3, POINTER(Dv), MAT4x4);
				}
				Dv = lagMul[sr + 2] * D(kconst->sp_j(), ib->toGlobal(kconst->f_i())) + lagMul[sr + 3] * D(kconst->sp_j(), ib->toGlobal(kconst->g_i()));
				m_lhs.plus(jc + 3, jc + 3, POINTER(Dv), MAT4x4);
				if (ib != &ground)
				{
					Bv = lagMul[sr + 2] * B(ib->getEP(), kconst->f_i()) + lagMul[sr + 3] * B(ib->getEP(), kconst->g_i());
					m_lhs.plus(jc + 0, ic + 3, POINTER(Bv), MAT3X4);
					Dv = lagMul[sr + 2] * transpose(B(jb->getEP(), kconst->sp_j()), B(ib->getEP(), kconst->f_i())) + lagMul[sr + 3] * transpose(B(jb->getEP(), kconst->sp_j()), B(ib->getEP(), kconst->g_i()));
					m_lhs.plus(jc + 3, ic + 3, POINTER(Dv), MAT4x4);
				}
			}
			sr += 5;
		}
	}
	for (massIterator m = md->pointMasses().begin(); m != md->pointMasses().end(); m++){
		mass* ms = m.value();
		if (!ms->ID())
			continue;
		size_t id = (ms->ID() - 1) * 7;
		MAT44D Dv;
		size_t sr = tdim - mdim;
		Dv = lagMul[sr - ms->ID()] * 2.0;
		m_lhs.plus(id + 3, id + 3, POINTER(Dv), MAT4x4);
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
	VEC3D v3, v3g, v3f, v;
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
			v = ib->toGlobal(kconst->g_i());
			rhs[kconst->startRow() + 3] = divbeta * v3.dot(ib->toGlobal(kconst->g_i()));
			rhs[kconst->startRow() + 4] = divbeta * v3.dot(ib->toGlobal(kconst->f_i()));
			break;
		case TRANSLATIONAL:
			v3 = jb->toGlobal(kconst->h_j());
			v3g = ib->toGlobal(kconst->g_i());
			v3f = ib->toGlobal(kconst->f_i());
			rhs[kconst->startRow() + 0] = divbeta * v3.dot(v3g);
			rhs[kconst->startRow() + 1] = divbeta * v3.dot(v3f);
			v3 = jb->getPosition() + jb->toGlobal(kconst->sp_j()) - ib->getPosition();
			rhs[kconst->startRow() + 2] = divbeta * v3.dot(v3g) - kconst->sp_i().dot(kconst->g_i());
			rhs[kconst->startRow() + 3] = divbeta * v3.dot(v3f) - kconst->sp_i().dot(kconst->f_i());
			rhs[kconst->startRow() + 4] = divbeta * v3g.dot(jb->toGlobal(kconst->g_j()));
			break;		}
	}
	for (dConstIterator it = md->drivingConstraints().begin(); it != md->drivingConstraints().end(); it++){
		drivingConstraint* dconst = it.value();
		size_t sr = dconst->startRow();
		rhs[sr] = divbeta * dconst->constraintEquation(ct);
	}
	int mr = tdim - (md->pointMasses().size()-1) - mdim;
	for (massIterator it = md->pointMasses().begin(); it != md->pointMasses().end(); it++){
		mass* m = it.value();
		if (m->ID() == 0)
			continue;
		double d = m->getEP().dot();
		rhs[mr++] = divbeta * (d - 1.0);
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
		//calcConstraintSystemJacobian(beta * dt * dt);
		for (int i(0); i < cjaco.nnz(); i++){
			lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
		}
		for (int i(0); i < mdim; i++) ee(i) -= pre(i);
		constraintEquation();
		e_norm = ee.norm();

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
	calcMassSystemJacobian(divalpha * beta * dt * dt);
	calcConstraintSystemJacobian(beta * dt * dt);
	for (int i(0); i < cjaco.nnz(); i++){
		lhs(cjaco.ridx[i], cjaco.cidx[i]) = lhs(cjaco.cidx[i], cjaco.ridx[i]) = cjaco.value[i];
	}
	for (int i(0); i < mdim; i++) ee(i) -= pre(i);
	constraintEquation();
	e_norm = ee.norm();

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
	qf_out.close();
	emit finished();
	return true;
}

bool mbd_simulation::gpuRun()
{
	return true;
}