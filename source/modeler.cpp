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
#include "glwidget.h"
#include <direct.h>
#include <QString>

modeler::modeler()
	: model_path("")
	, tsim(DEM)
	, ps(NULL)
{
	grav = VEC3F(0.f, -9.80665f, 0.f);
}

modeler::modeler(QString _name, tSimulation _sim, tUnit u, tGravity _dg)
	: model_path(_name)
	, tsim(_sim)
	, ps(NULL)
	, unit(u)
	, dg(_dg)
{
	switch (dg)
	{
	case PLUS_X: grav = VEC3F(9.80665f, 0.f, 0.f); break;
	case PLUS_Y: grav = VEC3F(0.f, 9.80665f, 0.f); break;
	case PLUS_Z: grav = VEC3F(0.f, 0.f, 9.80665f); break;
	case MINUS_X: grav = VEC3F(-9.80665f, 0.f, 0.f); break;
	case MINUS_Y: grav = VEC3F(0.f, -9.80665f, 0.f); break;
	case MINUS_Z: grav = VEC3F(0.f, 0.f, -9.80665f); break;
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
	if (ps) delete ps; ps = NULL;
}

cylinder* modeler::makeCylinder(QString _name, tMaterial _mat, tRoll _roll)
{
	cylinder cy(this, _name, _mat, _roll);
	cylinders[_name] = cy;
	objs[_name] = &(cylinders[_name]);
	return &(cylinders[_name]);
}

cube* modeler::makeCube(QString _name, tMaterial _mat, tRoll _roll)
{
 	cube cb(this, _name, _mat, _roll);
	cubes[_name] = cb;
//	cb.save_shape_data(io_model);
	objs[_name] = &(cubes[_name]);
	return &(cubes[_name]);
}

plane* modeler::makePlane(QString _name, tMaterial _mat, tRoll _roll)
{
 	plane pe(this, _name, _mat, _roll);
 	planes[_name] = pe;
// 	pe.save_shape_data(io_model);
	objs[_name] = &(planes[_name]);
 	return &(planes[_name]);
}

polygonObject* modeler::makePolygonObject(tImport tm, QString file)
{
	QFile qf(file);
	qf.open(QIODevice::ReadOnly);
	QTextStream qs(&qf);
	QString ch;
	polygonObject* _po;
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
				polygonObject po(this, file);
			//	po.define(tm, qs);
				pObjs[po.objectName()] = po;
				_po = &(pObjs[po.objectName()]);
				objs[po.objectName()] = _po;
				pObjs[po.objectName()].define(tm, qs);
			}
		}
		break;
	}
	qf.close();
	return _po;
}
	
// 	polygon po(this, _name, _mat, _roll);
// 	polygons[_name] = po;
// 	objs[_name] = &(polygons[_name]);
// 	return &(polygons[_name]);

kinematicConstraint* modeler::makeKinematicConstraint(QString _name, tKinematicConstraint kt, VEC3D& loc,
	mass* i, VEC3D& fi, VEC3D& gi,
	mass* j, VEC3D& fj, VEC3D& gj)
{
	kinematicConstraint* kin;
	switch (kt)
	{
	case REVOLUTE:
		kin = new revoluteConstraint(this, _name, kt, loc, i, fi, gi, j, fj, gj);
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
			cube c = cubes.find(_name).value();
			vector3<float> minp = c.min_point();
			vector3<float> maxp = c.max_point();
			VEC3D CuCenterp;
			CuCenterp.x = (maxp.x + minp.x) / 2;
			CuCenterp.y = (maxp.y + minp.y) / 2;
			CuCenterp.z = (maxp.z + minp.z) / 2;
			ms->setMassPoint(CuCenterp);
			break;
		}
		case PLANE:{
			plane pl= planes.find(_name).value();
			vector3<float> xw = pl.XW();
			vector3<float> w2 = pl.W2();
			vector3<float> w3 = pl.W3();
			vector3<float> w4 = pl.W4();
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
			QMap<QString, polygonObject>::iterator po = pObjs.find(_name);
			ms->setMassPoint(po.value().getOrigin());
// 			vector3<float> p = po.P();
// 			vector3<float> q = po.Q();
// 			vector3<float> r = po.R();
// 			VEC3D PgCenterp;
// 			PgCenterp.x = (p.x + q.x + r.x) / 3;
// 			PgCenterp.y = (p.y + q.y + r.y) / 3;
// 			PgCenterp.z = (p.z + q.z + r.z) / 3;
// 			ms->setMassPoint(PgCenterp);
			break;
		}
		case CYLINDER:{
			cylinder cy = cylinders.find(_name).value();
			VEC3D goc;
			ms->setPosition(cy.origin());
			ms->setEP(cy.orientation());
			//goc = cy.origin().To<double>();
			break;
		}
	}
	return ms;
}

particle_system* modeler::makeParticleSystem(QString _name)
{
	if (!ps)
		ps = new particle_system(_name, this);
	return ps;
}

collision* modeler::makeCollision(QString _name, float _rest, float _fric, float _rfric, tCollisionPair tcp, tContactModel tcm, void* o1, void* o2)
{
	collision* c = NULL;
	particle_system *ps;
	switch (tcp){
	case PARTICLES_PARTICLES:
		c = new collision_particles_particles(_name, this, (particle_system*)o1, tcm);
		c->setContactParameter(_rest, _fric, _rfric);
		break;
	case PARTICLES_PLANE:
		ps = (particle_system*)o1;
		c = new collision_particles_plane(_name, this, (particle_system*)o1, (plane*)o2, tcm);
		ps->addCollision(c);
		c->setContactParameter(_rest, _fric, _rfric);
		break;
	case PARTICLES_CYLINDER:
		ps = (particle_system*)o1;
		c = new collision_particles_cylinder(_name, this, ps, (cylinder*)o2, tcm);
		ps->addCollision(c);
		c->setContactParameter(_rest, _fric, _rfric);
		break;
	case PARTICLES_POLYGONOBJECT:
		ps = (particle_system*)o1;
		c = new collision_particles_polygonObject(_name, this, ps, (polygonObject*)o2, tcm);
		ps->addCollision(c);
		c->setContactParameter(_rest, _fric, _rfric);
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
	if (cubes.size())
	{
		foreach(cube value, cubes)
		{
			value.save_shape_data(ts);
		}
	}
	if (planes.size())
	{
		foreach(plane value, planes)
		{
			value.save_shape_data(ts);
		}
	}
	if (pObjs.size())
	{
		foreach (polygonObject value, pObjs)
		{
			value.save_shape_data(ts);
		}
	}
	if (cylinders.size())
	{
		foreach(cylinder value, cylinders)
		{
			value.save_shape_data(ts);
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
				VEC3F min_p, max_p;
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
				}
 			}
 			else if (ch == "PLANE") {
				int up;
				int tr, tmat;
				in >> id >> ch >> tr >> tmat >> up >> isExist;
				VEC3F p1, p2, p3, p4;
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
				}
 			}
			else if (ch == "POLYGON"){
				int up;
				int tr, tmat;
				in >> id >> ch >> tr >> tmat >> up >> isExist;
				in >> ch;
				polygonObject* po = makePolygonObject(NO_FORMAT, ch);
				po->setUpdate(up);
				gl->makePolygonObject(pObjs);
// 				VEC3F p, q, r;
// 				in >> p.x >> p.y >> p.z;
// 				in >> q.x >> q.y >> q.z;
// 				in >> r.x >> r.y >> r.z;
// 				polygon *po = makePolygon(ch, (tMaterial)tmat, (tRoll)tr);
// 				po->setUpdate((bool)up);
// 				po->define(p, q, r);
// 				po->setID(id);
// 				gl->makePolygon(po);
// 
				if (isExist){		//polygon 추가
					mass *m = makeMass(po->objectName());
					m->openData(in);
				}
			}
			else if (ch == "CYLINDER"){
				float br, tpr, len;
				int tr, tmat, up;
				VEC3F org, bpos, tpos;
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
				}
			}
 		}
 		else if (ch == "PARTICLES")
 		{
			unsigned int np;
			QString pfile, bo;
			float rho, E, pr, sh, rest, fric, rfric, coh;
			in >> ch >> np;
			in >> ch >> pfile;
			in >> ch >> bo;
			in >> ch >> rho;
			in >> ch >> E;
			in >> ch >> pr;
			//in >> ch >> sh;
			in >> ch >> rest;
			in >> ch >> sh;
			in >> ch >> fric;
			in >> ch >> rfric,
			in >> ch >> coh;
			makeParticleSystem("particles");
			ps->setParticlesFromFile(pfile, bo, np, rho, E, pr, sh);
			ps->setCollision(rest, fric, rfric, coh);
			gl->makeParticle(ps);
 		}
 		else if (ch == "COLLISION")
 		{
			float rest, fric, rfric;
			int tcm;
			QString nm, io, jo;
			in >> nm;
			in >> rest >> fric >> rfric >> tcm;
			in >> ch >> io;
			in >> ch >> jo;
	
			tCollisionPair cp = getCollisionPair(io != "particles" ? objs[io]->objectType() : ps->objectType(), jo != "particles" ? objs[jo]->objectType() : ps->objectType());
			if (io == "particles")
				makeCollision(nm, rest, fric, rfric, cp, (tContactModel)tcm, ps, objs[jo]);
			else if (jo == "particles")
				makeCollision(nm, rest, fric, rfric, cp, (tContactModel)tcm, ps, objs[io]);
			else
				makeCollision(nm, rest, fric, rfric, cp, (tContactModel)tcm, objs[io], objs[jo]);
 		}
 	}
	pf.close();
	unit = (tUnit)_unit;
	dg = (tGravity)_dg;
	switch (dg)
	{
	case PLUS_X: grav = VEC3F(9.80665f, 0.f, 0.f); break;
	case PLUS_Y: grav = VEC3F(0.f, 9.80665f, 0.f); break;
	case PLUS_Z: grav = VEC3F(0.f, 0.f, 9.80665f); break;
	case MINUS_X: grav = VEC3F(-9.80665f, 0.f, 0.f); break;
	case MINUS_Y: grav = VEC3F(0.f, -9.80665f, 0.f); break;
	case MINUS_Z: grav = VEC3F(0.f, 0.f, -9.80665f); break;
	}
}

void modeler::runExpression(float ct, float dt)
{
	if (objs.size()){
		foreach(object* value, objs)
		{
			if (value->expression())
				value->runExpression(ct, dt);
		}
	}
}

unsigned int modeler::numPolygonSphere()
{
	unsigned int _np = 0;
	if (pObjs.size())
	{
		foreach(polygonObject value, pObjs)
		{
			_np += value.numIndex();
		}
	}
	return _np;
}