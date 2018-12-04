#include "particleManager.h"
#include "model.h"
#include "glwidget.h"
//#include <QRandomGenerator>
#include <QList>

unsigned int particleManager::count = 0;

particleManager::particleManager()
	: pos(NULL)
	, pos_f(NULL)
	, np(0)
	, per_np(0)
	, per_time(0)
	, is_realtime_creating(false)
	, is_changed_particle(false)
	, one_by_one(false)
{
	obj = new object("particles", PARTICLES, PARTICLE);
}

particleManager::~particleManager()
{
	if (obj) delete obj; obj = NULL;
	if (pos) delete[] pos; pos = NULL;
	if (pos_f) delete[] pos_f; pos_f = NULL;
// 	if (mass) delete[] mass; mass = NULL;
// 	if (iner) delete[] iner; iner = NULL;
}

void particleManager::Save(QTextStream& qts)
{
	qts << endl
		<< "PARTICLES_DATA" << endl;
	foreach(QString log, logs)
	{
		qts << log;
	}
	qts << "END_DATA" << endl;
}

void particleManager::Open(QTextStream& qts)
{
	QString ch;
	while (ch != "END_DATA")
	{
		qts >> ch;
		if (ch == "CREATE_SHAPE")
			qts >> ch;
		if (ch == "cube")
		{
			QString n;
			int type;
			double dx, dy, dz;
			double lx, ly, lz;
			double spacing, min_radius, max_radius;
			double youngs, density, poisson, shear;
			qts >> ch >> n
				>> ch >> type
				>> ch >> dx >> dy >> dz
				>> ch >> lx >> ly >> lz
				>> ch >> spacing
				>> ch >> min_radius >> max_radius
				>> ch >> youngs >> density >> poisson >> shear;
			unsigned int _np = calculateNumCubeParticles(dx, dy, dz, min_radius, max_radius);
			CreateCubeParticle(
				n, (material_type)type, _np, dx, dy, dz, lx, ly, lz,
				spacing, min_radius, max_radius,
				youngs, density, poisson, shear);
		}
		else if (ch == "plane")
		{
			QString n;
			QString pfile;
			int type;
			int isr, obo;
			double dx, dz;
			unsigned int ny, _np, perNp;
			double pnt;
			double lx, ly, lz;
			double dirx, diry, dirz;
			double spacing, min_radius, max_radius;
			double youngs, density, poisson, shear;
			qts >> ch >> n
				>> ch >> type
				>> ch >> dx >> ny >> dz
				>> ch >> lx >> ly >> lz
				>> ch >> dirx >> diry >> dirz
				>> ch >> spacing
				>> ch >> min_radius >> max_radius
				>> ch >> isr >> _np >> perNp >> pnt >> obo >> pfile
				>> ch >> youngs >> density >> poisson >> shear;
			_np = calculateNumPlaneParticles(dx, ny, dz, min_radius, max_radius);
			CreatePlaneParticle(
				n, (material_type)type, dx, ny, dz, _np, lx, ly, lz,
				dirx, diry, dirz, spacing, min_radius, max_radius,
				youngs, density, poisson, shear, isr, perNp, pnt, obo, pfile);

		}
		else if (ch == "circle")
		{
			QString n;
			QString pfile;
			int type;
			int isr, obo;
			unsigned int nx, ny, _np, perNp;
			double dia;
			double pnt;
			double lx, ly, lz;
			double dx, dy, dz;
			double spacing, min_radius, max_radius;
			double youngs, density, poisson, shear;
			qts >> ch >> n
				>> ch >> type
				>> ch >> dia
				>> ch >> ny
				>> ch >> lx >> ly >> lz
				>> ch >> dx >> dy >> dz
				>> ch >> spacing
				>> ch >> min_radius >> max_radius
				>> ch >> isr >> _np >> perNp >> pnt >> obo >> pfile
				>> ch >> youngs >> density >> poisson >> shear;
			_np = calculateNumCircleParticles(dia, ny, min_radius, max_radius);
			CreateCircleParticle(
				n, (material_type)type, dia, _np, ny, lx, ly, lz,
				dx, dy, dz, spacing, min_radius, max_radius,
				youngs, density, poisson, shear, isr, perNp, pnt, obo, pfile);
		}
	}
}

unsigned int particleManager::NextCreatingPerGroup()
{
	if (np_group_iterator == np_group.end())
		return 0;
	unsigned int pn = *np_group_iterator;
	np_group_iterator++;
	return pn;
}

unsigned int particleManager::NextCreatingOne(unsigned int pn)
{
	//	t
	return 1;
}

QString particleManager::setParticleDataFromPart(QString& f)
{
	QFile qf(f);
	QString ret;
	qf.open(QIODevice::ReadOnly);
	double ct = 0.0;
	int precision = 0;
	int flag = 0;
	qf.read((char*)&ct, sizeof(double));
 	qf.read((char*)&precision, sizeof(int));
 	qf.read((char*)&flag, sizeof(int));
	if (np)
	{
		qf.read((char*)pos, sizeof(double) * np * 4);
		ret = QString("The position of particles is changed.(Time : %1, Np : %2)").arg(ct).arg(np);
		is_changed_particle = true;
	}
	else
	{
		//ret = QString("The number of particles is not matched.(Original : %1, New : %2)").arg(np).arg(np);
	}
	qf.close();
	return ret;
}

unsigned int particleManager::calculateNumPlaneParticles(double dx, unsigned int ny, double dz, double min_radius, double max_radius)
{
	VEC3UI ndim;
	
	double r = max_radius;
	double diameter = 2.0 * r;
	VEC3D dimension(dx, 0, dz);
	ndim = (dimension / diameter).To<unsigned int>() - VEC3UI(1, 0, 1);

	return ndim.x * ny * ndim.z;
}

unsigned int particleManager::calculateNumCircleParticles(double d, unsigned int ny, double min_radius, double max_radius)
{
	double r = max_radius;
	double cr = 0.5 * d;
	unsigned int nr = static_cast<unsigned int>(cr / (2.0 * r)) - 1;
	double rr = 2.0 * r * nr + r;
	double space = (cr - rr) / (nr + 1);
	unsigned int cnt = 1;
	for (unsigned int i = 1; i <= nr; i++)
	{
		double dth = (2.0 * r + space) / (i * (2.0 * r + space));
		unsigned int _np = static_cast<unsigned int>((2.0 * M_PI) / dth);
		cnt += _np;
	}
	return cnt * ny;
}

unsigned int particleManager::calculateNumCubeParticles(double dx, double dy, double dz, double min_radius, double max_radius)
{
	VEC3UI ndim;
	if (min_radius == max_radius)
	{
		double r = min_radius;
		double diameter = 2.0 * r;
		VEC3D dimension(dx, dy, dz);
		ndim = (dimension / diameter).To<unsigned int>() - VEC3UI(1, 1, 1);
	}
	return ndim.x * ndim.y * ndim.z;
}

VEC4D* particleManager::CreateCubeParticle(
	QString n, material_type type, unsigned int _np, double dx, double dy, double dz,
	double lx, double ly, double lz,
	double spacing, double min_radius, double max_radius,
	double youngs, double density, double poisson, double shear)
{

	particlesInfo pinfo;
	pinfo.sid = np;
	unsigned int pnp = np;
	pinfo.youngs = youngs;
	pinfo.density = density;
	pinfo.poisson = poisson;
	pinfo.shear = shear;
	pinfo.min_radius = min_radius;
	pinfo.max_radius = max_radius;
	pinfo.loc = VEC3D(lx, ly, lz);
	pinfo.dim = VEC3D(dx, dy, dz);
	pinfo.dir = 0;
	obj->setMaterial(type, youngs, density, poisson, shear);
	np += _np;
	pinfo.np = np - pinfo.sid;
	if (model::isSinglePrecision)
		pos_f = resizeMemory(pos_f, pnp, np);
	else
		pos = resizeMemory(pos, pnp, np);

	if (min_radius == max_radius)
	{
		double r = min_radius;
		double diameter = 2.0 * r;
		VEC3D dimension(dx, dy, dz);
		VEC3UI ndim = (dimension / diameter).To<unsigned int>() - VEC3UI(1,1,1);
		VEC3D plen = diameter * ndim.To<double>();
		VEC3D space = dimension - plen;
		space.x /= ndim.x + 1;
		space.y /= ndim.y + 1;
		space.z /= ndim.z + 1;
		VEC3D gab(diameter + space.x, diameter + space.y, diameter + space.z);
		VEC3D ran = r * space;
		unsigned int cnt = 0;
		for (unsigned int z = 0; z < ndim.z; z++)
		{
			double _z = lz + r + space.z;
// 			_z = lz + z * gab + frand() * ran;
			for (unsigned int y = 0; y < ndim.y; y++)
			{
				double _y = ly + r + space.y;
// 				_y = ly + y * gab + frand() * ran;
// 				if (_y + r > lmy) break;
				for (unsigned int x = 0; x < ndim.x; x++)
				{
					double _x = lx + r + space.x;
					
					VEC4D p = VEC4D(_x + x * gab.x + ran.x * frand(), _y + y * gab.y + ran.y * frand(), _z + z * gab.z + ran.z * frand(), r);
					if (model::isSinglePrecision)
						pos_f[pinfo.sid + cnt] = p.To<float>();
					else
						pos[pinfo.sid + cnt] = p;				
					cnt++;
				}
			}
		}
	}
	if (model::isSinglePrecision)
		GLWidget::GLObject()->makeParticle_f((float*)pos_f, np);
	else
		GLWidget::GLObject()->makeParticle((double*)pos, np);

	QString log;
	QTextStream qts(&log);
	qts << "CREATE_SHAPE " << "cube" << endl
		<< "NAME " << n << endl
		<< "MATERIAL_TYPE " << (int)type << endl
		<< "DIMENSION " << dx << " " << dy << " " << dz << endl
		<< "STARTPOINT " << lx << " " << ly << " " << lz << endl
		<< "SPACE " << spacing << endl
		<< "RADIUS " << min_radius << " " << max_radius << endl
		<< "MATERIAL " << youngs << " " << density << " " << poisson << " " << shear << endl;
		
	logs[n] = log;
	pinfos[n] = pinfo;
	count++;
	return pos;
}

VEC4D* particleManager::CreatePlaneParticle(
	QString n, material_type type, double dx, unsigned int ny, double dz, unsigned int _np,
	double lx, double ly, double lz,
	double dirx, double diry, double dirz,
	double spacing, double min_radius, double max_radius,
	double youngs, double density, double poisson, double shear, bool isr, unsigned int perNp, double pnt, bool obo, QString pf)
{
	particlesInfo pinfo;
	pinfo.sid = np;
	//pinfo.szp = 0;
	unsigned int pnp = np;
	pinfo.youngs = youngs;
	pinfo.density = density;
	pinfo.poisson = poisson;
	pinfo.shear = shear;
	pinfo.min_radius = min_radius;
	pinfo.max_radius = max_radius;
	pinfo.loc = VEC3D(lx, ly, lz);
	pinfo.dim = VEC3D(dx, 0.0, dz);
	pinfo.dir = VEC3D(dirx, diry, dirz);
	//np += nx * nz;
	np += _np;
	pinfo.np = np - pinfo.sid;
	obj->setMaterial(type, youngs, density, poisson, shear);
	if (model::isSinglePrecision)
		pos_f = resizeMemory(pos_f, pnp, np);
	else
		pos = resizeMemory(pos, pnp, np);

	double r = max_radius;
	double diameter = 2.0 * r;
	VEC3D dimension(dx, 0, dz);
	VEC3UI ndim = (dimension / diameter).To<unsigned int>() - VEC3UI(1, 0, 1);
	VEC3D plen = diameter * ndim.To<double>();
	VEC3D space = dimension - plen;
	space.x /= ndim.x + 1;
	//space.y /= ndim.y + 1;
	space.z /= ndim.z + 1;
	VEC3D gab(diameter + space.x, 0.0, diameter + space.z);
	VEC3D ran = 100.0 * r * space;
	unsigned int cnt = 0;
	unsigned int npg = 0;
	//np_group.push_back(cnt);
	for (unsigned int y = 0; y < ny; y++)
	{
		npg = 0;
		double _y = ly;
		for (unsigned int z = 0; z < ndim.z; z++)
		{			
			double _z = lz + space.z;
			for (unsigned int x = 0; x < ndim.x; x++)
			{
				double _x = lx + space.x;
				VEC4D p = VEC4D(_x + x * gab.x + ran.x * frand(), _y, _z + z * gab.z + ran.z * frand(), r);
				if (model::isSinglePrecision)
					pos_f[pinfo.sid + cnt] = p.To<float>();
				else
					pos[pinfo.sid + cnt] = p;
				cnt++;
				npg++;
			}
		}	
		np_group.push_back(npg);
	}
	QString pfile = "none";
	if (min_radius != max_radius)
	{
		double dr = max_radius - min_radius;
		srand(GetTickCount());
		for (unsigned int i = 0; i < np; i++)
		{
			pos[pinfo.sid + i].w = min_radius + dr * frand();
		}
		if (pf != "none")
		{
			pfile = pf;
			QFile qf(pfile);
			qf.open(QIODevice::ReadOnly);
			qf.read((char*)&_np, sizeof(unsigned int));
			qf.read((char*)pos + pinfo.sid, sizeof(double) * _np * 4);
			qf.close();
		}			
		else
		{
			pfile = model::path + model::name + "_init_position.par";
			QFile qf(pfile);
			qf.open(QIODevice::WriteOnly);
			qf.write((char*)&_np, sizeof(unsigned int));
			qf.write((char*)pos + pinfo.sid, sizeof(double) * _np * 4);
			qf.close();
		}			
	}
	per_time = pnt;
	is_realtime_creating = isr;
	one_by_one = obo;
	if (one_by_one)
		per_np = np / ndim.y;
	np_group_iterator = np_group.begin();

	if (model::isSinglePrecision)
		GLWidget::GLObject()->makeParticle_f((float*)pos_f, np);
	else
		GLWidget::GLObject()->makeParticle((double*)pos, np);
	QString log;
	QTextStream qts(&log);
	qts << "CREATE_SHAPE " << "plane" << endl
		<< "NAME " << n << endl
		<< "MATERIAL_TYPE " << (int)type << endl
		<< "DIMENSION " << dx << " " << ny << " " << dz << endl
		<< "STARTPOINT " << lx << " " << ly << " " << lz << endl
		<< "DIRECTION " << dirx << " " << diry << " " << dirz << endl
		<< "SPACE " << spacing << endl
		<< "RADIUS " << min_radius << " " << max_radius << endl
		<< "MAKING " << isr << " " << _np << " " << perNp << " " << pnt << " " << one_by_one << " " << pfile << endl
		<< "MATERIAL " << youngs << " " << density << " " << poisson << " " << shear << endl;
	pinfos[n] = pinfo;
	logs[n] = log;
	count++;
	return pos;
}

VEC4D* particleManager::CreateCircleParticle(
	QString n, material_type type, double cdia, unsigned int _np, unsigned int nh,
	double lx, double ly, double lz,
	double dx, double dy, double dz,
	double spacing, double min_radius, double max_radius,
	double youngs, double density, double poisson, double shear, bool isr /*= false*/, unsigned int perNp, double pnt, bool obo, QString pf)
{
	particlesInfo pinfo;
	//pinfo.szp = 0;
	pinfo.tp = 2;
	pinfo.sid = np;
	pinfo.cdia = cdia;
	unsigned int pnp = np;
	pinfo.youngs = youngs;
	pinfo.density = density;
	pinfo.poisson = poisson;
	pinfo.shear = shear;
	pinfo.min_radius = min_radius;
	pinfo.max_radius = max_radius;
	pinfo.loc = VEC3D(lx, ly, lz);
	pinfo.dim = VEC3D(0.0, 0.0, 0.0);
	pinfo.dir = VEC3D(dx, dy, dz);
	np += _np;
	pinfo.np = np - pinfo.sid;
	obj->setMaterial(type, youngs, density, poisson, shear);
	if (model::isSinglePrecision)
		pos_f = resizeMemory(pos_f, pnp, np);
	else
		pos = resizeMemory(pos, pnp, np);
	double r = max_radius;
	double cr = 0.5 * cdia;
	unsigned int nr = static_cast<unsigned int>(cr / (2.0 * r)) - 1;
	double rr = 2.0 * r * nr + r;
	double space = (cr - rr) / (nr + 1);
	unsigned int cnt = 0;
	unsigned int npg = 0;
	
	for (unsigned int y = 0; y < nh; y++)
	{
		npg = 0;
		pos[pinfo.sid + cnt] = VEC4D(lx, ly, lz, r);
		cnt++;
		for (unsigned int i = 1; i <= nr; i++)
		{
			double _r = i * (2.0 * r + space);
			double dth = (2.0 * r + space) / _r;
			unsigned int npr = static_cast<unsigned int>((2.0 * M_PI) / dth);
			dth = ((2.0 * M_PI) / npr);
			for (unsigned int j = 0; j < npr; j++)
			{
				VEC4D pp(lx + _r * cos(dth * j), ly, lz + _r * sin(dth * j), r);
				pos[pinfo.sid + cnt] = pp;
				cnt++;
				npg++;
			}
		}
		np_group.push_back(npg);
	}
	QString pfile = "none";
	if (min_radius != max_radius)
	{
		double dr = max_radius - min_radius;
		srand(GetTickCount());
		for (unsigned int i = 0; i < _np; i++)
		{
			pos[pinfo.sid + i].w = min_radius + dr * frand();
		}
		if (pf != "none")
		{
			pfile = pf;
			QFile qf(pfile);
			qf.open(QIODevice::ReadOnly);
			qf.read((char*)&_np, sizeof(unsigned int));
			qf.read((char*)pos + pinfo.sid, sizeof(double) * _np * 4);
			qf.close();
		}
		else
		{
			pfile = model::path + model::name + "_init_position.par";
			QFile qf(pfile);
			qf.open(QIODevice::WriteOnly);
			qf.write((char*)&_np, sizeof(unsigned int));
			qf.write((char*)pos + pinfo.sid, sizeof(double) * _np * 4);
			qf.close();
		}
	}
// 	if (min_radius == min_radius)
// 		r = min_radius;
// 	QList<VEC3D> pList;
// 	VEC3D pl;
// 	QList<unsigned int> iList;
// 	pList.push_back(pinfo.loc);
// 	int i = 1;
// 	unsigned int cnt = 0;
// 	iList.push_back(cnt++);
// 	
// 	while (1)
// 	{
// 		double _r = i * (2.0 * r + spacing);
// 		if (_r + r > 0.5 * cdia)
// 			break;
// 		double dth = (2 * r + spacing) / _r;
// 		unsigned int _np = static_cast<unsigned int>((2.0 * M_PI) / dth);
// 		dth = ((2.0 * M_PI) / _np);
// 		for (unsigned int j = 0; j < _np; j++)
// 		{
// 			pl = VEC3D(pinfo.loc.x + _r * cos(dth * j), pinfo.loc.y, pinfo.loc.z + _r * sin(dth * j));
// 			pList.push_back(pl);
// 			iList.push_back(cnt++);
// 		}
// 		i++;
// 	}
// // 	QRandomGenerator qran;
// // 	QList<unsigned int>::iterator iend = iList.end();
// // 	for (unsigned int i = 0; i < pList.size(); i++)
// // 	{
// // 		unsigned int ni = qran.bounded(pList.size() - 1);
// // 		QList<unsigned int>::iterator endIter = iList.begin() + i;
// // 		QList<unsigned int>::iterator isExist = qFind(iList.begin(), endIter, ni);
// // 		if (isExist != endIter)
// // 			continue;
// // 		iList.replace(i, ni);
// // 		iList.replace(ni, i);
// // 	}
// 	if (!isr) np += pList.size();
// 	else np = _np;
// 	pinfo.np = np - pinfo.sid;
// 
// 	if (model::isSinglePrecision)
// 		pos_f = resizeMemory(pos_f, pnp, np);
// 	else
// 		pos = resizeMemory(pos, pnp, np);
// 	
// 	if (!isr)
// 	{
// 		unsigned int cnt = 0;
// 		foreach(VEC3D p, pList)
// 		{
// 			pos[cnt] = VEC4D(p.x, p.y, p.z, r);
// 			cnt++;
// 		}
// 	}
// 	else
// 	{
// 		unsigned cnt = 0;
// 		unsigned int szp = pList.size();
// 		bool breaker = false;
// 		unsigned int k = 0;
// 		while (!breaker)
// 		{
// 			foreach(unsigned int i, iList)
// 			{
// 				VEC3D p = pList.at(i);
//  				double dc = (VEC3D(p.x, p.y, p.z) - VEC3D(lx, ly, lz)).length();
//  				double th = 0.5 * k * r / dc;
// 				VEC3D new_p = rotation_y(p, th);
// 				pos[pinfo.sid + cnt] = VEC4D(new_p.x, new_p.y, new_p.z, r);
// 				cnt++;
// 				if (cnt == _np)
// 				{
// 					breaker = true;
// 					break;
// 				}
// 			}
// 			k++;
// 		}
		//per_np = perNp;
	per_time = pnt;
	is_realtime_creating = isr;
	one_by_one = obo;
	if (one_by_one)
		per_np = perNp;
	else
		np_group_iterator = np_group.begin();
	//}
	if (model::isSinglePrecision)
		GLWidget::GLObject()->makeParticle_f((float*)pos_f, np);
	else
		GLWidget::GLObject()->makeParticle((double*)pos, np);

	QString log;
	QTextStream qts(&log);
	qts << "CREATE_SHAPE " << "circle" << endl
		<< "NAME " << n << endl
		<< "MATERIAL_TYPE " << (int)type << endl
		<< "DIAMETER " << cdia << endl
		<< "NUM_HEIGHT " << nh << endl
		<< "STARTPOINT " << lx << " " << ly << " " << lz << endl
		<< "DIRECTION " << dx << " " << dy << " " << dz << endl
		<< "SPACE " << spacing << endl
		<< "RADIUS " << min_radius << " " << max_radius << endl
		<< "MAKING " << isr << " " << _np << " " << perNp << " " << pnt << " " << one_by_one << " " << pfile << endl
		<< "MATERIAL " << youngs << " " << density << " " << poisson << " " << shear << endl;
 	pinfos[n] = pinfo;
 	logs[n] = log;
 	count++;
	return pos;
}

double particleManager::calcMass(double r)
{
	return obj->Density() * 4.0 * M_PI * pow(r, 3.0) / 3.0;
}
