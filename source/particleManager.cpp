#include "particleManager.h"
#include "model.h"
#include "glwidget.h"
#include <QRandomGenerator>
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
			unsigned int nx, ny, nz;
			double lx, ly, lz;
			double spacing, min_radius, max_radius;
			double youngs, density, poisson, shear;
			qts >> ch >> n
				>> ch >> type
				>> ch >> nx >> ny >> nz
				>> ch >> lx >> ly >> lz
				>> ch >> spacing
				>> ch >> min_radius >> max_radius
				>> ch >> youngs >> density >> poisson >> shear;
			CreateCubeParticle(
				n, (material_type)type, nx, ny, nz, lx, ly, lz,
				spacing, min_radius, max_radius,
				youngs, density, poisson, shear);
		}
		else if (ch == "plane")
		{
			QString n;
			int type;
			int isr, obo;
			unsigned int nx, ny, _np, perNp;
			double pnt;
			double lx, ly, lz;
			double dx, dy, dz;
			double spacing, min_radius, max_radius;
			double youngs, density, poisson, shear;
			qts >> ch >> n
				>> ch >> type
				>> ch >> nx >> ny
				>> ch >> lx >> ly >> lz
				>> ch >> dx >> dy >> dz
				>> ch >> spacing
				>> ch >> min_radius >> max_radius
				>> ch >> isr >> _np >> perNp >> pnt >> obo
				>> ch >> youngs >> density >> poisson >> shear;
			CreatePlaneParticle(
				n, (material_type)type, nx, ny, _np, lx, ly, lz,
				dx, dy, dz, spacing, min_radius, max_radius,
				youngs, density, poisson, shear, isr, perNp, pnt, obo);
		}
		else if (ch == "circle")
		{
			QString n;
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
				>> ch >> lx >> ly >> lz
				>> ch >> dx >> dy >> dz
				>> ch >> spacing
				>> ch >> min_radius >> max_radius
				>> ch >> isr >> _np >> perNp >> pnt >> obo
				>> ch >> youngs >> density >> poisson >> shear;
			CreateCircleParticle(
				n, (material_type)type, dia, _np, lx, ly, lz,
				dx, dy, dz, spacing, min_radius, max_radius,
				youngs, density, poisson, shear, isr, perNp, pnt, obo);
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

VEC4D* particleManager::CreateCubeParticle(
	QString n, material_type type, unsigned int nx, unsigned int ny, unsigned int nz,
	double lx, double ly, double lz,
	double spacing, double min_radius, double max_radius,
	double youngs, double density, double poisson, double shear
	)
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
	pinfo.dim = VEC3D(nx, ny, nz);
	pinfo.dir = 0;
	obj->setMaterial(type, youngs, density, poisson, shear);
	np += nx * ny * nz;
	pinfo.np = np - pinfo.sid;
	if (model::isSinglePrecision)
		pos_f = resizeMemory(pos_f, pnp, np);
	else
		pos = resizeMemory(pos, pnp, np);
	//mass = resizeMemory(mass, pnp, np);
// 	double r = 0.0;
// 	double dr = 0.0;
	if (min_radius == max_radius)
	{
		double r = min_radius;
		double gab = 2.0 * r + spacing;
		double ran = r * spacing * 100;
		unsigned int cnt = 0;
		for (unsigned int z = 0; z < nz; z++)
		{
			for (unsigned int y = 0; y < ny; y++)
			{
				for (unsigned int x = 0; x < nx; x++)
				{
					VEC4D p = VEC4D
						(lx + x * gab + frand() * ran,
						ly + y * gab + frand() * ran,
						lz + z * gab + frand() * ran,
						r);
					if (model::isSinglePrecision)
						pos_f[pinfo.sid + cnt] = p.To<float>();
					else
						pos[pinfo.sid + cnt] = p;
					cnt++;
				}
			}
		}
	}		
// 	else
// 	{
// 		double dr = max_radius - min_radius;
// 		double mx = 2.0 * max_radius * lx;
// 		double my = 2.0 * max_radius * ly;
// 		double mz = 2.0 * max_radius * lz;
// 		double x = lx, y = ly z = lz;
// 		double crad = 0;
// 		while (z + crad < mz)
// 		{
// 			while (y + crad < my)
// 			{
// 				while (x + crad < mx)
// 				{
// 					double prad = crad;
// 					crad = min_radius + dr * frand();
// 					VEC4D p = VEC4D
// 						(
// 						x + spacing + prad + crad,
// 						y + spacing + 
// 						)
// 				}
// 			}
// 		}
// 	}
		
	
// 	pos[0].x = 0.0;
// 	pos[0].z = 0.0;
// 	pos[1].x = 0.01;
// 	pos[1].z = 0.0;
	if (model::isSinglePrecision)
		GLWidget::GLObject()->makeParticle_f((float*)pos_f, np);
	else
		GLWidget::GLObject()->makeParticle((double*)pos, np);

	QString log;
	QTextStream qts(&log);
	qts << "CREATE_SHAPE " << "cube" << endl
		<< "NAME " << n << endl
		<< "MATERIAL_TYPE " << (int)type << endl
		<< "DIMENSION " << nx << " " << ny << " " << nz << endl
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
	QString n, material_type type, unsigned int nx, unsigned int nz, unsigned int _np,
	double lx, double ly, double lz,
	double dx, double dy, double dz,
	double spacing, double min_radius, double max_radius,
	double youngs, double density, double poisson, double shear, bool isr, unsigned int perNp, double pnt, bool obo)
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
	pinfo.dim = VEC3D(nx, 0.0, nz);
	pinfo.dir = VEC3D(dx, dy, dz);
	//np += nx * nz;
	np += _np;
	pinfo.np = np - pinfo.sid;
	obj->setMaterial(type, youngs, density, poisson, shear);
	if (model::isSinglePrecision)
		pos_f = resizeMemory(pos_f, pnp, np);
	else
		pos = resizeMemory(pos, pnp, np);

	double r = 0.0;
	if (min_radius == max_radius)
		r = min_radius;
	double gab = 2.0 * r + spacing;
	double ran = r * 0.001;
	unsigned int cnt = 0;
	double tv = M_PI / 180.0;
	if (!isr)
	{
		for (unsigned int z = 0; z < nz; z++)
		{
			for (unsigned int x = 0; x < nx; x++)
			{
				VEC3D p = VEC3D
					(
					x * gab + frand() * ran,
					0.0,
					z * gab + frand() * ran
					);
				VEC3D p3 = VEC3D(lx, ly, lz) + local2global_eulerAngle(VEC3D(tv * dx, tv * dy, tv * dz), p);
				if (model::isSinglePrecision)
					pos_f[cnt] = VEC4F((float)p3.x, (float)p3.y, (float)p3.z, (float)r);
				else
					pos[cnt] = VEC4D(p3.x, p3.y, p3.z, r);
				cnt++;
			}
		}
	}
	else
	{
		QList<VEC4D> pList;
		
		int i = 1;
		double dr = max_radius - min_radius;
		double x = lx, z = lz;
		double mx = 2.0 * max_radius * nx;
		double mz = 2.0 * max_radius * nz;
		
		double prad = 0;
		double crad = 0;
		unsigned int cnt = 0;
		bool breaker = false;
		unsigned int nparticlepergroup = 0;
		while (!breaker)
		{
			z = lz;
			unsigned int npg = 0;
			while (z + crad < mz)
			{				
				x = lx;
				crad = min_radius + dr * frand();
				pList.push_back(VEC4D(x, ly, z, crad));
				while (x + crad < mx)
				{
					prad = crad;
					crad = min_radius + dr * frand();
					x += prad + spacing + crad;
					double _gab = crad * 0.001 * frand();
					VEC4D pp = VEC4D(x + _gab, ly, z + _gab, crad);
					pos[pinfo.sid + cnt] = pp;
					cnt++;
					npg++;
					if (cnt == _np)
					{
						breaker = true;
						break;
					}			
				}
				if (breaker)
					break;
				z += 2.0 * max_radius + spacing;
			}
			np_group.push_back(npg);
		}
		per_time = pnt;
		is_realtime_creating = true;
		one_by_one = obo;
		if (one_by_one)
			per_np = perNp;
		np_group_iterator = np_group.begin();
// 		while (!breaker)
// 		{
// 			foreach(VEC4D p, pList)
// 			{
// 				double _gab = p.w * 0.001 * frand();
// 				pos[pinfo.sid + cnt] = VEC4D(p.x + _gab, p.y, p.z + _gab, p.w);
// 				cnt++;
// 				if (cnt == _np)
// 				{
// 					breaker = true;
// 					break;
// 				}
// 			}
// 		}
// 		per_time = pnt;
// 		is_realtime_creating = true;
// 		one_by_one = obo;
// 		if (!one_by_one)
// 			per_np = pList.size();
// 		else
// 			per_np = perNp;
	}
	if (model::isSinglePrecision)
		GLWidget::GLObject()->makeParticle_f((float*)pos_f, np);
	else
		GLWidget::GLObject()->makeParticle((double*)pos, np);
	QString log;
	QTextStream qts(&log);
	qts << "CREATE_SHAPE " << "plane" << endl
		<< "NAME " << n << endl
		<< "MATERIAL_TYPE " << (int)type << endl
		<< "DIMENSION " << nx << " " << nz << endl
		<< "STARTPOINT " << lx << " " << ly << " " << lz << endl
		<< "DIRECTION " << dx << " " << dy << " " << dz << endl
		<< "SPACE " << spacing << endl
		<< "RADIUS " << min_radius << " " << max_radius << endl
		<< "MAKING " << isr << " " << _np << " " << perNp << " " << pnt << " " << one_by_one << endl
		<< "MATERIAL " << youngs << " " << density << " " << poisson << " " << shear << endl;
	pinfos[n] = pinfo;
	logs[n] = log;
	count++;
	return pos;
}

VEC4D* particleManager::CreateCircleParticle(
	QString n, material_type type, double cdia, unsigned int _np,
	double lx, double ly, double lz,
	double dx, double dy, double dz,
	double spacing, double min_radius, double max_radius,
	double youngs, double density, double poisson, double shear, bool isr /*= false*/, unsigned int perNp, double pnt, bool obo)
{
	particlesInfo pinfo;
	//pinfo.szp = 0;
	pinfo.sid = np;
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
	obj->setMaterial(type, youngs, density, poisson, shear);
	double r = 0.0;
	if (min_radius == min_radius)
		r = min_radius;
	QList<VEC3D> pList;
	VEC3D pl;
	QList<unsigned int> iList;
	pList.push_back(pinfo.loc);
	int i = 1;
	unsigned int cnt = 0;
	iList.push_back(cnt++);
	
	while (1)
	{
		double _r = i * (2.0 * r + spacing);
		if (_r + r > 0.5 * cdia)
			break;
		double dth = (2 * r + spacing) / _r;
		unsigned int _np = static_cast<unsigned int>((2.0 * M_PI) / dth);
		dth = ((2.0 * M_PI) / _np);
		for (unsigned int j = 0; j < _np; j++)
		{
			pl = VEC3D(pinfo.loc.x + _r * cos(dth * j), pinfo.loc.y, pinfo.loc.z + _r * sin(dth * j));
			pList.push_back(pl);
			iList.push_back(cnt++);
		}
		i++;
	}
	QRandomGenerator qran;
	QList<unsigned int>::iterator iend = iList.end();
	for (unsigned int i = 0; i < pList.size(); i++)
	{
		unsigned int ni = qran.bounded(pList.size() - 1);
		QList<unsigned int>::iterator endIter = iList.begin() + i;
		QList<unsigned int>::iterator isExist = qFind(iList.begin(), endIter, ni);
		if (isExist != endIter)
			continue;
		iList.replace(i, ni);
		iList.replace(ni, i);
	}
	if (!isr) np += pList.size();
	else np = _np;
	pinfo.np = np - pinfo.sid;

	if (model::isSinglePrecision)
		pos_f = resizeMemory(pos_f, pnp, np);
	else
		pos = resizeMemory(pos, pnp, np);
	
	if (!isr)
	{
		unsigned int cnt = 0;
		foreach(VEC3D p, pList)
		{
			pos[cnt] = VEC4D(p.x, p.y, p.z, r);
			cnt++;
		}
	}
	else
	{
		unsigned cnt = 0;
		unsigned int szp = pList.size();
		bool breaker = false;
		unsigned int k = 0;
		while (!breaker)
		{
			foreach(unsigned int i, iList)
			{
				VEC3D p = pList.at(i);
 				double dc = (VEC3D(p.x, p.y, p.z) - VEC3D(lx, ly, lz)).length();
 				double th = 0.5 * k * r / dc;
				VEC3D new_p = rotation_y(p, th);
				pos[pinfo.sid + cnt] = VEC4D(new_p.x, new_p.y, new_p.z, r);
				cnt++;
				if (cnt == _np)
				{
					breaker = true;
					break;
				}
			}
			k++;
		}
		//per_np = perNp;
		per_time = pnt;
		is_realtime_creating = true;
		one_by_one = obo;
		if (!one_by_one)
			per_np = pList.size();
		else
			per_np = perNp;
	}
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
		<< "STARTPOINT " << lx << " " << ly << " " << lz << endl
		<< "DIRECTION " << dx << " " << dy << " " << dz << endl
		<< "SPACE " << spacing << endl
		<< "RADIUS " << min_radius << " " << max_radius << endl
		<< "MAKING " << isr << " " << _np << " " << perNp << " " << pnt << " " << one_by_one << endl
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
