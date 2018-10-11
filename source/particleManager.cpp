#include "particleManager.h"
#include "glwidget.h"

unsigned int particleManager::count = 0;

particleManager::particleManager()
	: pos(NULL)
	, np(0)
{
	obj = new object("particles", PARTICLES, PARTICLE);
}

particleManager::~particleManager()
{
	if (obj) delete obj; obj = NULL;
	if (pos) delete[] pos; pos = NULL;
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
			unsigned int nx, ny;
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
				>> ch >> youngs >> density >> poisson >> shear;
			CreatePlaneParticle(
				n, (material_type)type, nx, ny, lx, ly, lz,
				dx, dy, dz, spacing, min_radius, max_radius,
				youngs, density, poisson, shear);
		}
	}
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
	pos = resizeMemory(pos, pnp, np);
	//mass = resizeMemory(mass, pnp, np);
	double r = 0.0;
	if (min_radius == min_radius)
		r = min_radius;
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
				pos[pinfo.sid + cnt] = p;
				cnt++;
			}
		}
	}
// 	pos[0].x = 0.0;
// 	pos[0].z = 0.0;
// 	pos[1].x = 0.01;
// 	pos[1].z = 0.0;
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
	QString n, material_type type, unsigned int nx, unsigned int nz,
	double lx, double ly, double lz,
	double dx, double dy, double dz,
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
	pinfo.dim = VEC3D(nx, 0.0, nz);
	pinfo.dir = VEC3D(dx, dy, dz);
	np += nx * nz;
	pinfo.np = np - pinfo.sid;
	obj->setMaterial(type, youngs, density, poisson, shear);
	pos = resizeMemory(pos, pnp, np);

	double r = 0.0;
	if (min_radius == min_radius)
		r = min_radius;
	double gab = 2.0 * r + spacing;
	double ran = r * 0.001;
	unsigned int cnt = 0;
	double tv = M_PI / 180.0;
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
			pos[cnt] = VEC4D(p3.x, p3.y, p3.z, r);
			cnt++;
		}
	}

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
