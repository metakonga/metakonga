#include "geometryObjects.h"
#include "glwidget.h"
#include "Objects.h"

geometryObjects::geometryObjects()
{

}

geometryObjects::~geometryObjects()
{
	qDeleteAll(markers);
}

void geometryObjects::Save(QTextStream& qts)
{
	qts << endl
		<< "GEOMETRY_OBJECTS_DATA" << endl;
	foreach(QString log, logs)
	{
		qts << log;
	}
	qts << "END_DATA" << endl;	
}

void geometryObjects::Open(QTextStream& qts)
{
	QString ch;
	while (ch != "END_DATA")
	{
		qts >> ch;
		if (ch == "OBJECT")
			qts >> ch;
		
		if (ch == "cube")
		{
			QString _name;
			int mt, gu;
			double e, p, d, s;
			VEC3D start, end;
			qts >> ch >> _name
				>> ch >> mt
				>> ch >> gu
				>> ch >> start.x >> start.y >> start.z
				>> ch >> end.x >> end.y >> end.z
				>> ch >> e >> p >> d >> s;
			makeCube(
				_name, (material_type)mt, (geometry_use)gu,
				start, end, e, p, d, s);
		}
		else if (ch == "plane")
		{
			QString _name;
			int mt, gu;
			double e, p, d, s;
			VEC3D pa, pb, pc, pd;
			qts >> ch >> _name
				>> ch >> mt
				>> ch >> gu
				>> ch >> pa.x >> pa.y >> pa.z
				>> ch >> pb.x >> pb.y >> pb.z
				>> ch >> pc.x >> pc.y >> pc.z
				>> ch >> pd.x >> pd.y >> pd.z
				>> ch >> e >> p >> d >> s;
			makePlane(
				_name, (material_type)mt, (geometry_use)gu,
				pa, pb, pc, pd, e, p, d, s);
		}
		else if (ch == "polygon")
		{
			QString _name, file;
			QString tt;
			double v;
			int mt, gu, ist;
			double e, p, d, s;
			VEC3D loc;
			qts >> ch >> _name >> ch;
			file = qts.readLine();
			file.remove(0, 1);
			qts //>> ch >> file
				>> ch >> loc.x >> loc.y >> loc.z
				>> ch >> mt
				>> ch >> gu
				>> ch >> ist
				>> ch >> e >> p >> d >> s
				>> ch >> tt >> ch >> v;
			vpolygon* vp = GLWidget::GLObject()->makePolygonObject(_name, (import_shape_type)ist, file, loc.x, loc.y, loc.z);
			if (tt == "poly_refinement" && v)
				polyRefinement(ch, v);// ->splitTriangle(v);
			makePolygonObject(
				_name, (geometry_use)gu, file, (import_shape_type)ist,loc,
				vp->NumTriangles(), vp->VertexList(), vp->IndexList(),
				(material_type)mt, e, p, d, s);
		}
	}
}

void geometryObjects::insertObject(object* _obj)
{
	objs[_obj->Name()] = _obj;
	markers.push_back(_obj);
	//pms.push_back(dynamic_cast<pointMass*>(_obj));
}

object* geometryObjects::Object(QString n)
{
	QStringList l = objs.keys();
	QStringList::const_iterator it = qFind(l, n);
	if (it == l.end())
		return NULL;
	return objs[n];
}

QString geometryObjects::polyRefinement(QString n, double v)
{
	vpolygon* vpoly = dynamic_cast<vpolygon*>(GLWidget::GLObject()->Object(n));
	unsigned int pre_ntri = vpoly->NumTriangles();
	if (vpoly)
		vpoly->splitTriangle(v);
	QString log = "TOOL poly_refinement " + vpoly->name() + " " + QString("%1").arg(v) + "\n";
	logs[vpoly->name() + "_tool"] = log;
	unsigned int post_ntri = vpoly->NumTriangles();
	return "The refinement of " + vpoly->name() + " is finished.\nThe number of triangles is increased from " + QString("%1").arg(pre_ntri) + " to " + QString("%1").arg(post_ntri);
}

cube* geometryObjects::makeCube(
	QString _name, material_type mt, geometry_use gu, 
	VEC3D& start, VEC3D& end, double e, double p, double d, double s)
{
	cube* c = new cube(_name, gu);
	c->define(start, end);
	c->setMaterial(mt, e, d, p, s);
	objs[_name] = c;
	markers.push_back(c);
	GLWidget::GLObject()->makeCube(c);
//	database::DB()->addChild(database::CUBE_ROOT, c->Name());

	QString log;
	QTextStream qts(&log);
	qts << "OBJECT " << "cube" << endl
		<< "NAME " << _name << endl
		<< "MATERIAL_TYPE " << (int)mt << endl
		<< "GEOMETRY_USE " << (int)gu << endl
		<< "START_POINT " << start.x << " " << start.y << " " << start.z << endl
		<< "END_POINT " << end.x << " " << end.y << " " << end.z << endl
		<< "MATERIAL " << e << " " << p << " " << d << " " << s << endl;
	logs[c->Name()] = log;
	return c;
}

plane* geometryObjects::makePlane(
	QString _name, material_type mt, geometry_use gu, 
	VEC3D& pa, VEC3D& pb, VEC3D& pc, VEC3D& pd, 
	double e, double p, double d, double s)
{
	plane* pl = new plane(_name, gu);
	pl->define(pa, pb, pc, pd);
	pl->setMaterial(mt, e, d, p, s);
	objs[_name] = pl;
	markers.push_back(pl);
	GLWidget::GLObject()->makePlane(pl);
//	database::DB()->addChild(database::PLANE_ROOT, pl->Name());

	QString log;
	QTextStream qts(&log);
	qts << "OBJECT " << "plane" << endl
		<< "NAME " << _name << endl
		<< "MATERIAL_TYPE " << (int)mt << endl
		<< "GEOMETRY_USE " << (int)gu << endl
		<< "POINT_A " << pa.x << " " << pa.y << " " << pa.z << endl
		<< "POINT_B " << pb.x << " " << pb.y << " " << pb.z << endl
		<< "POINT_C " << pc.x << " " << pc.y << " " << pc.z << endl
		<< "POINT_D " << pd.x << " " << pd.y << " " << pd.z << endl
		<< "MATERIAL " << e << " " << p << " " << d << " " << s << endl;
	logs[pl->Name()] = log;
	return pl;
}

polygonObject* geometryObjects::makePolygonObject
	(QString nm, geometry_use gu, QString file, import_shape_type t,
	VEC3D& loc, unsigned int ntri, double* vertexList, unsigned int* indexList,
	material_type mt, double e, double p, double d, double s)
{
	polygonObject* po = new polygonObject(nm, gu);
	po->setMaterial(mt, e, d, p, s);
	po->define(t, loc, ntri, vertexList, indexList);
	objs[nm] = po;
	markers.push_back(po);
	pobjs[polygonObject::Number()] = po;
//	database::DB()->addChild(database::POLYGON_ROOT, po->Name());

	QString log;
	QTextStream qts(&log);
	qts << "OBJECT " << "polygon" << endl
		<< "NAME " << nm << endl
		<< "FILE " << file << endl
		<< "CM " << loc.x << " " << loc.y << " " << loc.z << endl
		<< "MATERIAL_TYPE " << (int)mt << endl
		<< "GEOMETRY_USE " << (int)gu << endl
		<< "IMPORT_SHAPE_TYPE " << (int)t << endl
		<< "MATERIAL " << e << " " << p << " " << d << " " << s << endl;
	QStringList l = logs.keys();
	QString tg = nm + "_tool";
	QStringList::const_iterator it = qFind(l, nm + "_tool");
	if (it != l.end())
	{
		qts << logs[tg] << endl;
		logs.take(tg);
	}
	else
	{
		qts << "TOOL none none 0" << endl;
	}
	logs[po->Name()] = log;
	return po;
}