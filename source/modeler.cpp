#include "modeler.h"
#include "cube.h"
#include "mass.h"
#include "plane.h"
#include "polygonObject.h"
#include "cylinder.h"
#include "collision_particles_particles.h"
#include "collision_particles_plane.h"
#include "collision_particles_cylinder.h"
#include "collision_particles_polygonObject.h"
#include "revoluteConstraint.h"
#include "translationalConstraint.h"
#include "drivingConstraint.h"
#include "glwidget.h"
#include "database.h"
#include <direct.h>
#include <QString>
#include <QPlainTextEdit>

modeler::modeler()
	: model_path("")
	, tsim(DEM)
	, ps(NULL)
	, ncube(0)
	, nplane(0)
	, ncylinder(0)
	, npoly(0)
{
	grav = VEC3D(0.f, -9.80665f, 0.f);
}

modeler::modeler(QString _name, tSimulation _sim, tUnit u, tGravity _dg)
	: model_path(_name)
	, tsim(_sim)
	, ps(NULL)
	, unit(u)
	, dg(_dg)
	, ncube(0)
	, nplane(0)
	, ncylinder(0)
	, npoly(0)
{
	switch (dg)
	{
	case PLUS_X: grav = VEC3D(9.80665f, 0.f, 0.f); break;
	case PLUS_Y: grav = VEC3D(0.f, 9.80665f, 0.f); break;
	case PLUS_Z: grav = VEC3D(0.f, 0.f, 9.80665f); break;
	case MINUS_X: grav = VEC3D(-9.80665f, 0.f, 0.f); break;
	case MINUS_Y: grav = VEC3D(0.f, -9.80665f, 0.f); break;
	case MINUS_Z: grav = VEC3D(0.f, 0.f, -9.80665f); break;
	}
	//grav = VEC3F(0.f, -9.80665f, 0.f);
	int begin = model_path.lastIndexOf("/");
	name = model_path.mid(begin + 1);
	QString model_file = model_path + "/" + name + ".mde";
	QFile check_model(model_file);
	check_model.open(QIODevice::ReadOnly);
	if (!check_model.isOpen()){
		_mkdir(model_path.toStdString().c_str());
	}
	check_model.close();
}

modeler::~modeler()
{
	QMapIterator<QString, collision*> _cs(cs);
	while (_cs.hasNext()){
		_cs.next();
		delete _cs.value();
	}
	QMapIterator<object*, mass*> _m(masses);
	while (_m.hasNext()){
		_m.next();
		delete _m.value();
	}
	QMapIterator<QString, kinematicConstraint*> kin(consts);
	while (kin.hasNext()){
		kin.next();
		delete kin.value();
	}
	qDeleteAll(objs);
	qDeleteAll(dconsts);
	if (ps) delete ps; ps = NULL;
}

cylinder* modeler::makeCylinder(QString _name, tMaterial _mat, tRoll _roll)
{
	cylinder *cy = new cylinder(this, _name, _mat, _roll);
	objs[_name] = cy;
	ncylinder++;
	db->addChild(database::CYLINDER_ROOT, _name);
	return cy;
}

cube* modeler::makeCube(QString _name, tMaterial _mat, tRoll _roll)
{
 	cube *cb = new cube(this, _name, _mat, _roll);
	objs[_name] = cb;
	ncube++;
	db->addChild(database::CUBE_ROOT, _name);
	return cb;
}

plane* modeler::makePlane(QString _name, tMaterial _mat, tRoll _roll)
{
 	plane *pe = new plane(this, _name, _mat, _roll);
  	objs[_name] = pe;
	nplane++;
	db->addChild(database::PLANE_ROOT, _name);
	return pe;
}

polygonObject* modeler::makePolygonObject(tImport tm, QString file)
{
	QFile qf(file);
	qf.open(QIODevice::ReadOnly);
	QTextStream qs(&qf);
	QString ch;
	polygonObject* po = NULL;
	if (tm == NO_FORMAT)
	{
		qs >> ch >> ch;
		if (ch == "MilkShape")
			tm = MILKSHAPE_3D_ASCII;
	}
	switch (tm){
	case MILKSHAPE_3D_ASCII:
		{
			int nmesh = 0;
			while (!qs.atEnd())
			{
				qs >> ch;
				if (ch == "Meshes:"){
					qs >> nmesh;
					break;
				}
			}
			for (int i = 0; i < nmesh; i++)
			{
				po = new polygonObject(this, file);
			//	po.define(tm, qs);
				//pObjs[po.objectName()] = po;
				//_po = &(pObjs[po.objectName()]);
				po->define(tm, qs);
				objs[po->objectName()] = po;
				polygons.push_back(po);
				npoly++;
				//pObjs[po.objectName()].define(tm, qs);
			}
		}
		break;
	}
	qf.close();
	db->addChild(database::POLYGON_ROOT, po->objectName());
	return po;
}
	
// 	polygon po(this, _name, _mat, _roll);
// 	polygons[_name] = po;
// 	objs[_name] = &(polygons[_name]);
// 	return &(polygons[_name]);

kinematicConstraint* modeler::makeKinematicConstraint(QString _name, tKinematicConstraint kt,
	mass* i, VEC3D& spi, VEC3D& fi, VEC3D& gi,
	mass* j, VEC3D& spj, VEC3D& fj, VEC3D& gj)
{
	kinematicConstraint* kin;
	switch (kt)
	{
	case REVOLUTE:
		kin = new revoluteConstraint(this, _name, kt, i, spi, fi, gi, j, spj, fj, gj);
		break;
	case TRANSLATIONAL:
		kin = new translationalConstraint(this, _name, kt, i, spi, fi, gi, j, spj, fj, gj);
		break;
	}
	consts[_name] = kin;
	return kin;
}

mass* modeler::makeMass(QString _name)
{
	QMap<QString, object*>::iterator it = objs.find(_name);
	object* obj = NULL;
	mass *ms = new mass(this, _name);
	if (it == objs.end()){
		masses[NULL] = ms;
		return ms;
	}
	else{
		obj = it.value();
		masses[obj] = ms;
		obj->addPointMass(ms);
	}
		
	switch (obj->objectType()){				//calculate Mass Center Position
		case CUBE:{
			cube *c = getChildObject<cube*>(_name);
			VEC3D minp = c->min_point();
			VEC3D maxp = c->max_point();
			VEC3D CuCenterp;
			CuCenterp.x = (maxp.x + minp.x) / 2;
			CuCenterp.y = (maxp.y + minp.y) / 2;
			CuCenterp.z = (maxp.z + minp.z) / 2;
			ms->setMassPoint(CuCenterp);
			break;
		}
		case PLANE:{
			plane *pl = getChildObject<plane*>(_name);// .find(_name).value();
			VEC3D xw = pl->XW();
			VEC3D w2 = pl->W2();
			VEC3D w3 = pl->W3();
			VEC3D w4 = pl->W4();
			VEC3D PlMidp1;
			PlMidp1.x = (xw.x + w2.x) / 2;
			PlMidp1.y = (xw.y + w2.y) / 2;
			PlMidp1.z = (xw.z + w2.z) / 2;
			VEC3D PlMidp2;
			PlMidp2.x = (w3.x + w4.x) / 2;
			PlMidp2.y = (w3.y + w4.y) / 2;
			PlMidp2.z = (w3.z + w4.z) / 2;
			VEC3D PlCenterp;
			PlCenterp.x = (PlMidp1.x + PlMidp2.x) / 2;
			PlCenterp.y = (PlMidp1.y + PlMidp2.y) / 2;
			PlCenterp.z = (PlMidp1.z + PlMidp2.z) / 2;
			ms->setMassPoint(PlCenterp);
			break;
		}
		case POLYGON:{
			polygonObject* pobj = getChildObject<polygonObject*>(_name);// <QString, polygonObject>::iterator po = pObjs.find(_name);
			ms->setMassPoint(pobj->getOrigin());
			break;
		}
		case CYLINDER:{
			cylinder *cy = getChildObject<cylinder*>(_name);// .find(_name).value();
			VEC3D goc;
			ms->setPosition(cy->origin());
			ms->setEP(cy->orientation());
			//goc = cy.origin().To<double>();
			break;
		}
	}
	db->addChild(database::MASS_ROOT, _name);
	return ms;
}

particle_system* modeler::makeParticleSystem(QString _name)
{
	if (!ps)
		ps = new particle_system(_name, this);
	return ps;
}

drivingConstraint* modeler::makeDrivingConstraint(QString _name, kinematicConstraint* kconst, tDriving td, double val)
{
	drivingConstraint* dc = new drivingConstraint(_name);
	dc->define(kconst, td, val);
	dconsts[_name] = dc;
	return dc;
}

collision* modeler::makeCollision(QString _name, double _rest, double _fric, double _rfric, double _coh, double _ratio, tCollisionPair tcp, tContactModel tcm, void* o1, void* o2)
{
	collision* c = NULL;
	switch (tcp){
	case PARTICLES_PARTICLES:
		c = new collision_particles_particles(_name, this, ps, tcm);
		c->setContactParameter(
			ps->youngs(), ps->youngs(), ps->poisson(), ps->poisson(),
			ps->shear(), ps->shear(), _rest, _fric, _rfric, _coh, _ratio);
		break;
	case PARTICLES_PLANE:{
		plane *pl = (plane*)o2;
		c = new collision_particles_plane(_name, this, ps, pl, tcm);
	//	ps->addCollision(c);
		c->setContactParameter(
			ps->youngs(), pl->youngs(), ps->poisson(), pl->poisson(),
			ps->shear(), pl->shear(), _rest, _fric, _rfric, _coh, _ratio);
		break;
	}
	case PARTICLES_CYLINDER:{
		cylinder *cy = (cylinder*)o2;
		c = new collision_particles_cylinder(_name, this, ps, cy, tcm);
	//	ps->addCollision(c);
		c->setContactParameter(
			ps->youngs(), cy->youngs(), ps->poisson(), cy->poisson(),
			ps->shear(), cy->shear(), _rest, _fric, _rfric, _coh, _ratio);
		break;
	}
	case PARTICLES_POLYGONOBJECT:{
		polygonObject *po = (polygonObject*)o2;
		c = new collision_particles_polygonObject(_name, this, ps, po, tcm);
		//ps->addCollision(c);
		c->setContactParameter(
			ps->youngs(), po->youngs(), ps->poisson(), po->youngs(),
			ps->shear(), po->shear(), _rest, _fric, _rfric, _coh, _ratio);
		break;
	}
	}


	if (c)
		cs[_name] = c;

	return c;
}

void modeler::saveModeler()
{
	QString model_file = model_path + "/" + name + ".mde";
	QFile io_model(model_file);
	io_model.open(QIODevice::WriteOnly);
	QTextStream ts(&io_model);
	ts << "model_path " << model_path << endl;
	ts << "sim_type " << tsim << endl;
	ts << "name " << name << endl;
	ts << "unit " << (int)unit << endl;
	ts << "gravity_direction " << (int)dg << endl;
	ts << "gravity " << grav.x << " " << grav.y << " " << grav.z << endl;
	if (objs.size())
	{
		foreach(object* value, objs)
		{
			value->save_object_data(ts);
		}
	}

	if (ps){
		if (ps->numParticle())
		{
			ps->saveParticleSystem(io_model);
		}
	}
	
	if (cs.size())
	{
		foreach(collision* value, cs)
		{
			value->save_collision_data(ts);
		}
	}
	io_model.close();
	//comm("Model file was saved - " + model_file);
}

void modeler::openModeler(GLWidget *gl, QString& file)
{
	QFile pf(file);
	pf.open(QIODevice::ReadOnly);
	QTextStream in(&pf);
	QString ch;
	int ts;
	int _unit;
	int _dg;
	int isExist = false;
	unsigned int id;
	while (!in.atEnd())
	{
		in >> ch;
		if (ch == "model_path") in >> model_path;
		else if (ch == "name") in >> name;
		else if (ch == "unit") in >> _unit;
		else if (ch == "gravity_direction") in >> _dg;
		else if (ch == "sim_type") in >> ts;
		else if (ch == "gravity") in >> grav.x >> grav.y >> grav.z;
//		break;
		else if (ch == "OBJECT")
		{
 			in >> ch;
 			if (ch == "CUBE") {
				int up;
				int tr, tmat;
				in >> id >> ch >> tr >> tmat >> up >> isExist;
				VEC3D min_p, max_p;
				in >> min_p.x >> min_p.y >> min_p.z;
				in >> max_p.x >> max_p.y >> max_p.z;
				cube *c = makeCube(ch, (tMaterial)tmat, (tRoll)tr);
				c->setUpdate((bool)up);
				c->setID(id);
				c->define(min_p, max_p);
				gl->makeCube(c);
				if (isExist){
					mass *m = makeMass(c->objectName());
					m->openData(in);
					m->setBaseGeometryType(CUBE);
					m->setGeometryObject(gl->getVObjectFromName(m->name()));
				}
 			}
 			else if (ch == "PLANE") {
				int up;
				int tr, tmat;
				in >> id >> ch >> tr >> tmat >> up >> isExist;
				VEC3D p1, p2, p3, p4;
				in >> p1.x >> p1.y >> p1.z;
				in >> p2.x >> p2.y >> p2.z;
				in >> p3.x >> p3.y >> p3.z;
				in >> p4.x >> p4.y >> p4.z;
				plane *p = makePlane(ch, (tMaterial)tmat, (tRoll)tr);
				p->setUpdate((bool)up);
				p->setID(id);
				p->define(p1, p2, p3, p4);
				gl->makePlane(p);
				
				if (isExist){		//plane추가
					mass *m = makeMass(p->objectName());
					m->openData(in);
					m->setBaseGeometryType(PLANE);
					m->setGeometryObject(gl->getVObjectFromName(m->name()));
				}
 			}
			else if (ch == "POLYGON"){
				int up;
				int tr, tmat;
				in >> id >> ch >> tr >> tmat >> up >> isExist;
				in >> ch;
				polygonObject* po = makePolygonObject(NO_FORMAT, ch);
				po->setUpdate(up);
				polygons.push_back(po);
				gl->makePolygonObject(po);

				if (isExist){		//polygon 추가
					mass *m = makeMass(po->objectName());
					m->openData(in);
					m->setBaseGeometryType(POLYGON);
					m->setPolyGeometryObject(gl->getVPolyObjectFromName(m->name()));
				}
			}
			else if (ch == "CYLINDER"){
				double br, tpr, len;
				int tr, tmat, up;
				VEC3D org, bpos, tpos;
				in >> id >> ch >> tr >> tmat >> up >> isExist;
				in >> br >> tpr >> len;
				in >> org.x >> org.y >> org.z;
				in >> bpos.x >> bpos.y >> bpos.z;
				in >> tpos.x >> tpos.y >> tpos.z;
				cylinder *cy = makeCylinder(ch, (tMaterial)tmat, (tRoll)tr);
				cy->setUpdate((bool)up);
				cy->setID(id);
				cy->define(br, tpr, bpos, tpos);
				gl->makeCylinder(cy);
				if (isExist){
					mass *m = makeMass(cy->objectName());
					m->openData(in);
					m->setBaseGeometryType(CYLINDER);
					m->setGeometryObject(gl->getVObjectFromName(m->name()));
				}
			}
 		}
 		else if (ch == "PARTICLES")
 		{
			unsigned int np;
			QString pfile, bo;
			double rho, E, pr, sh;// rest, fric, rfric, coh, sratio;
			in >> ch >> np;
			in >> ch >> pfile;
			in >> ch >> bo;
			in >> ch >> rho;
			in >> ch >> E;
			in >> ch >> pr;
			in >> ch >> sh;
			makeParticleSystem("particles");
			ps->setParticlesFromFile(pfile, bo, np, rho, E, pr, sh);
			//ps->setCollision(rest, fric, rfric, coh, sratio);
			gl->makeParticle(ps);
		}
		else if (ch == "STACK")
		{
			int tg;
			unsigned int nStack, npPerStack;
			double interval;
			in >> tg;
			in >> ch >> nStack;
			in >> ch >> interval;
			in >> ch >> npPerStack;
			ps->setGenerationMethod((tGenerationParticleMethod)tg, nStack, interval, npPerStack);
		}
		else if (ch == "CLUSTER")
		{
			int consist = 0;
			unsigned int nc = 0;
			in >> ch >> consist;
			ps->setParticleCluster(consist);
			//particle_cluster::setConsistNumber(consist);
			//ps->setParticleCluster(consist);
		}
 		else if (ch == "COLLISION")
 		{
			double rest, fric, rfric, coh, sratio;
			int tcm;
			QString nm, io, jo;
			in >> nm;
			in >> rest >> fric >> rfric >> coh >> sratio >> tcm;
			in >> ch >> io;
			in >> ch >> jo;
	
			tCollisionPair cp = getCollisionPair(io != "particles" ? objs[io]->objectType() : ps->objectType(), jo != "particles" ? objs[jo]->objectType() : ps->objectType());
			if (io == "particles" && jo != "particles")
				makeCollision(nm, rest, fric, rfric, coh, sratio, cp, (tContactModel)tcm, ps, objs[jo]);
			else if (io != "particles" && jo == "particles")
				makeCollision(nm, rest, fric, rfric, coh, sratio, cp, (tContactModel)tcm, ps, objs[io]);
			else if (io == "particles" && jo == "particles")
				makeCollision(nm, rest, fric, rfric, coh, sratio, cp, (tContactModel)tcm, ps, ps);
			else
				makeCollision(nm, rest, fric, rfric, coh, sratio, cp, (tContactModel)tcm, objs[io], objs[jo]);
 		} 
 	}
	pf.close();
	unit = (tUnit)_unit;
	dg = (tGravity)_dg;
	switch (dg)
	{
	case PLUS_X: grav = VEC3D(9.80665f, 0.f, 0.f); break;
	case PLUS_Y: grav = VEC3D(0.f, 9.80665f, 0.f); break;
	case PLUS_Z: grav = VEC3D(0.f, 0.f, 9.80665f); break;
	case MINUS_X: grav = VEC3D(-9.80665f, 0.f, 0.f); break;
	case MINUS_Y: grav = VEC3D(0.f, -9.80665f, 0.f); break;
	case MINUS_Z: grav = VEC3D(0.f, 0.f, -9.80665f); break;
	}
}

void modeler::runExpression(double ct, double dt)
{
	if (objs.size()){
		foreach(object* value, objs)
		{
			if (value->expression())
				value->runExpression(ct, dt);
		}
	}
	if (dconsts.size()){
		foreach(drivingConstraint* value, dconsts)
		{
			value->driving(ct);
		}
	}
}

unsigned int modeler::numPolygonSphere()
{
	unsigned int _np = 0;
	if (npoly)
	{
		foreach(object* value, objs)
		{
			if (value->objectType() == POLYGON)
				_np += getChildObject<polygonObject*>(value->objectName())->numIndex();
		}
	}
	return _np;
}

// void modeler::comm(QString com)
// {
// 	pte->appendPlainText(com);
// }

void modeler::actionDelete(const QString& tg)
{
	object* obj = objs.take(tg);
	if(obj)
		delete obj;
}
