#include "mbd_model.h"
#include "GLWidget.h"
#include "pointMass.h"
#include "forceElement.h"
#include "fixedConstraint.h"
#include "revoluteConstraint.h"
#include "sphericalConstraint.h"
#include "translationalConstraint.h"
#include "axialRotationForce.h"
#include "universalConstraint.h"
#include "springDamperModel.h"
#include "modelManager.h"
#include <direct.h>
#include <QString>

mbd_model::mbd_model()
	: model()
	, mbd_model_name("")
	, ground(NULL)
	, start_time_simulation(0)
	, is2D(false)
{
	ground = new pointMass(QString("ground"));
	ground->setMassType(pointMass::GROUND);
	//GLWidget::GLObject()->makeMarker("ground", ground->Position());
	ground->setID(-1);
}

mbd_model::mbd_model(QString _name)
	: model()
	, mbd_model_name(_name)
	, ground(NULL)
	, is2D(false)
	, start_time_simulation(0)
{
	ground = new pointMass(QString("ground"));
	ground->setMassType(pointMass::GROUND);
	//GLWidget::GLObject()->makeMarker("ground",  ground->Position());
	ground->setID(-1);
}

mbd_model::~mbd_model()
{
	if (ground) delete ground; ground = NULL;
	//qDeleteAll(masses);
	qDeleteAll(consts);
	qDeleteAll(forces);
}

kinematicConstraint* mbd_model::createKinematicConstraint(
	QString _name, kinematicConstraint::Type kt,
	pointMass* i, VEC3D& spi, VEC3D& fi, VEC3D& gi,
	pointMass* j, VEC3D& spj, VEC3D& fj, VEC3D& gj)
{
	kinematicConstraint* kin;
	switch (kt)
	{
	case kinematicConstraint::FIXED:
		kin = new fixedConstraint(this, _name, kt, i, spi, fi, gi, j, spj, fj, gj);
		break;
	case kinematicConstraint::REVOLUTE:
		kin = new revoluteConstraint(this, _name, kt, i, spi, fi, gi, j, spj, fj, gj);
		break;
	case kinematicConstraint::SPHERICAL:
		kin = new sphericalConstraint(this, _name, kt, i, spi, fi, gi, j, spj, fj, gj);
		break;
	case kinematicConstraint::TRANSLATIONAL:
		kin = new translationalConstraint(this, _name, kt, i, spi, fi, gi, j, spj, fj, gj);
		break;
	case kinematicConstraint::UNIVERSAL:
		kin = new universalConstraint(this, _name, kt, i, spi, fi, gi, j, spj, fj, gj);
		break;
	}
	consts[_name] = kin;
//	database::DB()->addChild(database::CONSTRAINT_ROOT, _name);

	QString log;
	QTextStream qts(&log);
	qts << "ELEMENT " << "constraint" << endl
		<< "NAME " << _name << endl
		<< "TYPE " << kt << endl
		<< "FIRST_BODY " << i->Name() << endl
		<< "SECOND_BODY " << j->Name() << endl
		<< "FIRST_JOINT_COORDINATE "
		<< spi.x << " " << spi.y << " " << spi.z << " " 
		<< fi.x << " " << fi.y << " " << fi.z << " " 
		<< gi.x << " " << gi.y << " " << gi.z << endl
		<< "FIRST_JOINT_COORDINATE "
		<< spj.x << " " << spj.y << " " << spj.z << " " 
		<< fj.x << " " << fj.y << " " << fj.z << " " 
		<< gj.x << " " << gj.y << " " << gj.z << endl;
	other_logs[_name] = log;
	return kin;
}

kinematicConstraint* mbd_model::createKinematicConstraint(QTextStream& qts)
{
	QString ch;
	QString kin_name;
	QString base_name, action_name;
	VEC3D loc;
	VEC3D spi, fi, gi;
	VEC3D spj, fj, gj;
	int tp;
	qts >> ch >> kin_name >> ch >> tp
		>> ch >> loc.x >> loc.y >> loc.z;
	qts >> ch >> base_name
		>> ch >> spi.x >> spi.y >> spi.z
		>> ch >> fi.x >> fi.y >> fi.z
		>> ch >> gi.x >> gi.y >> gi.z
		>> ch >> action_name
		>> ch >> spj.x >> spj.y >> spj.z
		>> ch >> fj.x >> fj.y >> fj.z
		>> ch >> gj.x >> gj.y >> gj.z;
	pointMass *base = base_name == "ground" ? ground : masses[base_name];
	pointMass *action = action_name == "ground" ? ground : masses[action_name];
	kinematicConstraint* kin =  createKinematicConstraint(
		kin_name, (kinematicConstraint::Type)tp,
		base, spi, fi, gi,
		action, spj, fj, gj);
	kin->setLocation(loc);
	return kin;
}

// kinematicConstraint* mbd_model::createCableConstraint(QString _name, pointMass* fi, VEC3D& fspi, pointMass* fj, VEC3D& fspj, pointMass* si, VEC3D& sspi, pointMass* sj, VEC3D& sspj)
// {
// // 	cableConstraint *cable = new cableConstraint(
// // 		_name, fi, fspi, fj, fspj, si, sspi, sj, sspj);
// // 	cables[_name] = cable;
// // 	consts[_name] = cable;
// // 	VEC3D sp = fi->Position() + fi->toGlobal(fspi);
// // 	VEC3D ep = fj->Position() + fj->toGlobal(fspj);
// // 	//GLWidget::GLObject()->createLine(_name + "_f", sp.x, sp.y, sp.z, ep.x, ep.y, ep.z);
// // 
// // 	sp = si->Position() + si->toGlobal(sspi);
// // 	ep = sj->Position() + sj->toGlobal(sspj);
// // 	//GLWidget::GLObject()->createLine(_name + "_s", sp.x, sp.y, sp.z, ep.x, ep.y, ep.z);
// // 	return cable;
// }

// kinematicConstraint* mbd_model::createCableConstraint(QTextStream& qts)
// {
// 	QString pass;
// 	QString c_name;
// 	QString fi_name, fj_name, si_name, sj_name;
// 	VEC3D fspi, fspj, sspi, sspj;
// 	double fc = 0;
// 	double sc = 0;
// 	qts >> pass >> c_name
// 		>> pass >> fi_name
// 		>> pass >> fj_name
// 		>> pass >> si_name
// 		>> pass >> sj_name
// 		>> pass >> fspi.x >> fspi.y >> fspi.z
// 		>> pass >> fspj.x >> fspj.y >> fspj.z
// 		>> pass >> sspi.x >> sspi.y >> sspi.z
// 		>> pass >> sspj.x >> sspj.y >> sspj.z
// 		>> pass >> fc >> pass >> sc;
// 	pointMass *fi = fi_name == "ground" ? ground : masses[fi_name];
// 	pointMass *fj = fj_name == "ground" ? ground : masses[fj_name];
// 	pointMass *si = si_name == "ground" ? ground : masses[si_name];
// 	pointMass *sj = sj_name == "ground" ? ground : masses[sj_name];
// 	return createCableConstraint(c_name, fi, fspi, fj, fspj, si, sspi, sj, sspj);
// }
// 
// kinematicConstraint* mbd_model::createGearConstraint(QString _name, pointMass* i, kinematicConstraint* ik, pointMass* j, kinematicConstraint* jk, double r)
// {
// // 	gearConstraint *gear = new gearConstraint(_name, i, ik, j, jk, r);
// // 	gear->setGroundPointer(ground);
// // 	gears[_name] = gear;
// // 	consts[_name] = gear;
// // 	return gear;
// }

springDamperModel* mbd_model::createSpringDamperElement(
	QString _name,
	pointMass* i, VEC3D& bLoc, 
	pointMass* j, VEC3D& aLoc, 
	double k, double c)
{
	springDamperModel* fe = new springDamperModel(_name, this, i, bLoc, j, aLoc, k, c);
	forces[_name] = fe;
	//database::DB()->addChild(database::SPRING_DAMPER_ROOT, _name);

	QString log;
	QTextStream qts(&log);
	qts << "ELEMENT " << "tsda" << endl
		<< "NAME " << _name << endl
		<< "FIRST_BODY " << i->Name() << endl
		<< "SECOND_BODY " << j->Name() << endl
		<< "FIRST_LOCATION " << bLoc.x << " " << bLoc.y << " " << bLoc.z << endl
		<< "SECOND_LOCATION " << aLoc.x << " " << aLoc.y << " " << aLoc.z << endl
		<< "COEFF_SPRING " << k << endl << "COEFF_DAMPING " << c << endl;
	other_logs[_name] = log;
	return fe;
}

springDamperModel* mbd_model::createSpringDamperElement(QTextStream& qts)
{
	QString ch, nm;
	QString i_name, j_name;
	VEC3D iLoc, jLoc, iLocal, jLocal;
	double init_l, k, c;
	qts >> ch >> nm
		>> ch >> i_name
		>> ch >> j_name
		>> ch >> iLoc.x >> iLoc.y >> iLoc.z
		>> ch >> jLoc.x >> jLoc.y >> jLoc.z
		>> ch >> iLocal.x >> iLocal.y >> iLocal.z
		>> ch >> jLocal.x >> jLocal.y >> jLocal.z
		>> ch >> init_l >> k >> c;
	pointMass *fi = i_name == "ground" ? ground : masses[i_name];
	pointMass *fj = j_name == "ground" ? ground : masses[j_name];
	return createSpringDamperElement(nm, fi, iLoc, fj, jLoc, k, c);
}

axialRotationForce* mbd_model::createAxialRotationForce(
	QString _name, pointMass* i, pointMass* j, VEC3D loc, VEC3D u, double v)
{
	axialRotationForce* arf = new axialRotationForce(_name, this, loc, u, i, j);
	arf->setForceValue(v);
	forces[_name] = arf;
	return arf;
}

axialRotationForce* mbd_model::createAxialRotationForce(QTextStream& qts)
{
	QString ch, nm;
	QString i_name, j_name;
	VEC3D loc, u;
	double fv = 0.0;
	qts >> ch >> nm
		>> ch >> i_name
		>> ch >> j_name
		>> ch >> loc.x >> loc.y >> loc.z
		>> ch >> u.x >> u.y >> u.z
		>> ch >> fv;
	pointMass *fi = i_name == "ground" ? ground : masses[i_name];
	pointMass *fj = j_name == "ground" ? ground : masses[j_name];
	return createAxialRotationForce(nm, fi, fj, loc, u, fv);
	//arf->setForceValue(fv);
}

drivingConstraint* mbd_model::createDrivingConstraint(
	QString _nm, kinematicConstraint* _kin, drivingConstraint::Type _tp, double iv, double cv)
{
	drivingConstraint *dc = new drivingConstraint(_nm);
	dc->define(_kin, _tp, iv, cv);
	drivings[_nm] = dc;
	QString log;
	QTextStream qts(&log);
	qts << "ELEMENT " << "driving_constraint" << endl
		<< "NAME " << _nm << endl
		<< "TARGET_JOINT " << _kin->name() << endl
		<< "DRIVING_TYPE " << (int)_tp << endl
		<< "INITIAL_VALUE " << iv << endl
		<< "CONSTANT_VALUE " << cv << endl;
		//<< "STARTING_TIME " << 
	other_logs[_nm] = log;
	return dc;
}

// contactPair* mbd_model::createContactPair(
// 	QString _nm, pointMass* ib, pointMass* jb)
// {
// 	contactPair* cp = new contactPair(_nm);
// 	cp->setFirstBody(ib);
// 	cp->setSecondBody(jb);
// 	cpairs[_nm] = cp;
// 	return cp;
// }

void mbd_model::set2D_Mode(bool b)
{
	is2D = b;
}

bool mbd_model::mode2D()
{
	return is2D;
}

pointMass* mbd_model::PointMass(QString nm)
{
	QStringList l = masses.keys();
	QStringList::const_iterator it = qFind(l, nm);
	if (it == l.end())
		return NULL;
	return masses[nm];
}

kinematicConstraint* mbd_model::kinConstraint(QString nm)
{
	QStringList l = consts.keys();
	QStringList::const_iterator it = qFind(l, nm);
	if (it == l.end())
		return NULL;
	return consts[nm];
}

void mbd_model::insertPointMass(pointMass* pm)
{
	masses[pm->Name()] = pm;
	//database::DB()->addChild(database::RIGID_BODY_ROOT, pm->Name());
	VEC3D p = pm->Position();
	EPD ep = pm->getEP();
	VEC3D piner = pm->DiagonalInertia();
	VEC3D siner = pm->SymetricInertia();
	pm->setViewMarker(GLWidget::GLObject()->makeMarker(pm->Name(), p));
	QString log;
	QTextStream qts(&log);
	qts << "ELEMENT " << "poly_mass" << endl
		<< "NAME " << pm->Name() << endl
		<< "MASS " << pm->Mass() << endl
		<< "MATERIAL_TYPE " << pm->MaterialType() << endl
		<< "POSITION " << p.x << " " << p.y << " " << p.z << endl
		<< "PARAMETER " << ep.e0 << " " << ep.e1 << " " << ep.e2 << " " << ep.e3 << endl
		<< "D_INERTIA " << piner.x << " " << piner.y << " " << piner.z << endl
		<< "S_INERTIA " << siner.x << " " << siner.y << " " << siner.z << endl;
	body_logs[pm->Name()] = log;
}

pointMass* mbd_model::createPointMass(
	QString _name, double mass, VEC3D piner, VEC3D siner, 
	VEC3D p, EPD ep /*= EPD(1.0, 0.0, 0.0, 0.0)*/)
{
	pointMass* rb = new pointMass(_name);
	rb->setMass(mass);
	rb->setDiagonalInertia(piner.x, piner.y, piner.z);
	rb->setSymetryInertia(siner.x, siner.y, siner.z);
	rb->setPosition(p);
	rb->setEP(ep);
	rb->setViewMarker(GLWidget::GLObject()->makeMarker(_name, p));
	masses[_name] = rb;
	modelManager::MM()->GeometryObject()->addMarkerObject(rb);
	//database::DB()->addChild(database::RIGID_BODY_ROOT, _name);
	
	QString log;
	QTextStream qts(&log);
	qts << "ELEMENT " << "rigid" << endl
		<< "NAME " << _name << endl
		<< "MASS " << mass << endl
		<< "MATERIAL_TYPE " << rb->MaterialType() << endl
		<< "POSITION " << p.x << " " << p.y << " " << p.z << endl
		<< "PARAMETER " << ep.e0 << " " << ep.e1 << " " << ep.e2 << " " << ep.e3 << endl
		<< "D_INERTIA " << piner.x << " " << piner.y << " " << piner.z << endl
		<< "S_INERTIA " << siner.x << " " << siner.y << " " << siner.z << endl;
	body_logs[_name] = log;

	return rb;
}

pointMass* mbd_model::Ground()
{
	return ground;
}

void mbd_model::Open(QTextStream& qts)
{
	QString ch;
	//int tp = 0;
	while (ch != "END_DATA")
	{
		qts >> ch;
		if (ch == "ELEMENT")
			qts >> ch;
		if (ch == "point_mass")
		{
			QString _name;
			int mt;
			double mass;
			VEC3D p, piner, siner;
			EPD ep;
			qts >> ch >> _name
				>> ch >> mass
				>> ch >> mt
				>> ch >> p.x >> p.y >> p.z
				>> ch >> ep.e0 >> ep.e1 >> ep.e2 >> ep.e3
				>> ch >> piner.x >> piner.y >> piner.z
				>> ch >> siner.x >> siner.y >> siner.z;
			pointMass* pm = NULL;
			if (modelManager::MM()->GeometryObject())
				pm = dynamic_cast<pointMass*>(modelManager::MM()->GeometryObject()->Object(_name));
			if(!pm)
				createPointMass(_name, mass, piner, siner, p, ep);
			else
			{
				pm->setMass(mass);
				pm->setPosition(p);
				pm->setEP(ep);
				pm->setDiagonalInertia(piner.x, piner.y, piner.z);
				pm->setSymetryInertia(siner.x, siner.y, siner.z);
				insertPointMass(pm);
				vobject* vo = GLWidget::GLObject()->Object(_name);
				if (vo)
					pm->setViewObject(vo);
				pm->updateView(p, ep2e(ep));
			}
				
		}
// 		if ((pointMass::Type)tp == pointMass::POLYMER)
// 		{
// 			QString _name;
// 			int mt;
// 			double mass;
// 			VEC3D p, piner, siner;
// 			EPD ep;
// 			qts >> ch >> _name
// 				>> ch >> mass
// 				>> ch >> mt
// 				>> ch >> p.x >> p.y >> p.z
// 				>> ch >> ep.e0 >> ep.e1 >> ep.e2 >> ep.e3
// 				>> ch >> piner.x >> piner.y >> piner.z
// 				>> ch >> siner.x >> siner.y >> siner.z;
// 			pointMass* pm = dynamic_cast<pointMass*>(modelManager::MM()->GeometryObject()->Object(_name));
// 			pm->setMass(mass);
// 			pm->setPosition(p);
// 			pm->setEP(ep);
// 			pm->setDiagonalInertia(piner.x, piner.y, piner.z);
// 			pm->setSymetryInertia(siner.x, siner.y, siner.z);
// 			GLWidget::GLObject()->Objects()[pm->Name()]->setInitialPosition(pm->Position());
// 			insertPointMass(pm);
// 		//	createRigidBody(_name, mass, piner, siner, p, ep);
// 		}
		else if (ch == "constraint")
		{
			QString _name, ib, jb;
			int kt;
			VEC3D spi, fi, gi, spj, fj, gj;
			qts >> ch >> _name >> ch >> kt >> ch >> ib >> ch >> jb
				>> ch >> spi.x >> spi.y >> spi.z >> fi.x >> fi.y >> fi.z >> gi.x >> gi.y >> gi.z
				>> ch >> spj.x >> spj.y >> spj.z >> fj.x >> fj.y >> fj.z >> gj.x >> gj.y >> gj.z;
			createKinematicConstraint(
				_name, (kinematicConstraint::Type)kt,
				(ib == "ground" ? ground : masses[ib]), spi, fi, gi, 
				(jb == "ground" ? ground : masses[jb]), spj, fj, gj);
		}
		else if (ch == "drive_constraint")
		{
			QString _name, target;
			int tp;
			double iv, cv, st;
			qts >> ch >> _name 
				>> ch >> tp 
				>> ch >> target
				>> ch >> st 
				>> ch >> iv >> cv;
			kinematicConstraint *kc = kinConstraint(target);
			drivingConstraint *dc = createDrivingConstraint(_name, kc, (drivingConstraint::Type)tp, iv, cv);
			dc->setStartTime(st);

		}
		else if (ch == "tsda")
		{
			QString _name, ib, jb;
			double k, c;
			VEC3D bLoc, aLoc;
			qts >> ch >> _name >> ch >> ib >> ch >> jb
				>> ch >> bLoc.x >> bLoc.y >> bLoc.z
				>> ch >> aLoc.x >> aLoc.y >> aLoc.z
				>> ch >> k >> ch >> c;
			createSpringDamperElement(_name, masses[ib], bLoc, masses[jb], aLoc, k, c);
		}
	}
}

void mbd_model::Save(QTextStream& qts)
{
	qts << endl
		<< "MULTIBODY_MODEL_DATA " << mbd_model_name << endl;
	foreach(pointMass* pm, masses)
		pm->saveData(qts);

	foreach(kinematicConstraint* kc, consts)
		kc->saveData(qts);
// 	foreach(cableConstraint* cc, cables)
// 	{
// 		cc->saveCableConstraintData(qts);
// 	}
	foreach(forceElement* fe, forces)
		fe->saveData(qts);
	foreach(drivingConstraint* dc, drivings)
		dc->saveData(qts);
// 	foreach(QString log, body_logs)
// 	{
// 		qts << log;
// 	}
// 	foreach(QString log, other_logs)
// 	{
// 		qts << log;
// 	}
	qts << "END_DATA" << endl;
}

void mbd_model::exportPointMassResultData2TXT()
{
	foreach(pointMass* pm, masses)
	{
		pm->exportResultData2TXT();
	}
	
}

void mbd_model::exportReactionForceResultData2TXT()
{
	foreach(kinematicConstraint* kc, consts)
	{
		kc->exportResultData2TXT();
	}
}

void mbd_model::runExpression(double ct, double dt)
{
// 	if (objs.size()){
// 		foreach(object* value, objs)
// 		{
// 			if (value->expression())
// 				value->runExpression(ct, dt);
// 		}
// 	}
// 	if (dconsts.size()){
// 		foreach(drivingConstraint* value, dconsts)
// 		{
// 			value->driving(ct);
// 		}
// 	}
}

void mbd_model::updateCableInitLength()
{
// 	foreach(cableConstraint* cc, cables)
// 	{
// 		cc->updateCableInitLength();
// 	}
}

void mbd_model::updateAndInitializing()
{
// 	memset(rhs)
// 	foreach()
}

void mbd_model::setStartTimeForSimulation(double sts)
{
	start_time_simulation = sts;
}

// QMap<QString, v3epd_type> mbd_model::setStartingData(startingModel* stm)
// {
// 	QMap<QString, v3epd_type> out;
// 	foreach(pointMass* pm, masses)
// 	{
// 		resultStorage::pointMassResultData pmrd = stm->MBD_BodyData()[pm->Name()];
// 		pm->setPosition(pmrd.pos);
// 		pm->setVelocity(pmrd.vel);
// 		pm->setAcceleration(pmrd.acc);
// 		pm->setEP(pmrd.ep);
// 		VEC4D ev = 0.5 * transpose(pmrd.ep.G(), pmrd.omega);
// 		pm->setEV(EPD(ev.x, ev.y, ev.z, ev.w));
// 		pm->setEA(pmrd.ea);
// 		out[pm->Name()] = v3epd_type{ pmrd.pos, pmrd.ep };
// 		pm->makeTransformationMatrix();
// 	}
// 	return out;
// }

// void mbd_model::setPointMassDataFromStartingModel(QMap<int, resultStorage::pointMassResultData>& d)
// {
// 
// }

// unsigned int mbd_model::numPolygonSphere()
// {
// 	unsigned int _np = 0;
// 	if (npoly)
// 	{
// 		foreach(object* value, objs)
// 		{
// 			if (value->objectType() == POLYGON)
// 				_np += getChildObject<polygonObject*>(value->objectName())->numIndex();
// 		}
// 	}
// 	return _np;
// }

// void mbd_model::comm(QString com)
// {
// 	pte->appendPlainText(com);
// }

// void mbd_model::actionDelete(const QString& tg)
// {
// 	object* obj = objs.take(tg);
// 	if (obj)
// 		delete obj;
// }
