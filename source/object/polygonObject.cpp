#include "polygonObject.h"
#include "modeler.h"
#include "mphysics_cuda_dec.cuh"
#include "mass.h"

polygonObject::polygonObject()
	: object()
	, h_poly(NULL)
	, d_poly(NULL)
	, h_sph(NULL)
	, d_sph(NULL)
	, h_mass(NULL)
	, d_mass(NULL)
	, vertice(NULL)
	, indice(NULL)
	, nvertex(0)
	, nindex(0)
	, maxRadii(0)
{

}

polygonObject::polygonObject(modeler *_md, QString file)
	: object(_md, QString("pobj"), POLYGON, ACRYLIC, ROLL_BOUNDARY)
	, h_poly(NULL)
	, d_poly(NULL)
	, h_sph(NULL)
	, d_sph(NULL)
	, vertice(NULL)
	, h_mass(NULL)
	, d_mass(NULL)
	, indice(NULL)
	, nvertex(0)
	, nindex(0)
	, maxRadii(0)
	, filePath(file)
{
	int begin = file.lastIndexOf("/") + 1;
	int end = file.lastIndexOf(".");
	QString _nm = file.mid(begin, end - begin);
	name = _nm;
}

polygonObject::polygonObject(const polygonObject& _poly)
	: object(_poly)
	, h_poly(NULL)
	, d_poly(NULL)
	, h_sph(NULL)
	, d_sph(NULL)
	, vertice(NULL)
	, indice(NULL)
	, h_mass(NULL)
	, d_mass(NULL)
	, nvertex(_poly.numVertex())
	, nindex(_poly.numIndex())
	, maxRadii(_poly.maxRadius())
	, filePath(_poly.meshDataFile())
	, org(_poly.getOrigin())
{
	if (_poly.hostMassInfo())
	{
		h_mass = new host_polygon_mass_info;
		memcpy(h_mass, _poly.hostMassInfo(), sizeof(host_polygon_mass_info));
	}
	if (_poly.hostPolygonInfo())
	{
		h_poly = new host_polygon_info[nindex];
		memcpy(h_poly, _poly.hostPolygonInfo(), sizeof(host_polygon_info) * nindex);
	}
	if (_poly.hostSphereSet())
	{
		h_sph = new VEC4D[nindex];
		memcpy(h_sph, _poly.hostSphereSet(), sizeof(VEC4D) * nindex);
	}
	if (_poly.vertexSet())
	{
		vertice = new VEC3D[nvertex];
		memcpy(vertice, _poly.vertexSet(), sizeof(VEC3D) * nvertex);
	}
	if (_poly.indexSet())
	{
		indice = new VEC3UI[nindex];
		memcpy(indice, _poly.indexSet(), sizeof(VEC3UI) * nindex);
	}
// 	if (_poly.devicePolygonInfo())
// 	{
// 		checkCudaErrors(cudaMalloc((void**)&d_poly, sizeof(device_polygon_info) * nindex));
// 		checkCudaErrors(cudaMemcpy(d_poly,  _poly.devicePolygonInfo(), sizeof(device_polygon_info) * nindex, cudaMemcpyDeviceToDevice));
// 	}
// 	if (_poly.deviceSphereSet())
// 	{
// 		checkCudaErrors(cudaMalloc((void**)&d_sph, sizeof(VEC4D) * nindex));
// 		checkCudaErrors(cudaMemcpy(d_sph, _poly.deviceSphereSet(), sizeof(VEC4D) * nindex, cudaMemcpyDeviceToDevice));
// 	}
}

polygonObject::~polygonObject()
{
	if (vertice) delete[] vertice; vertice = NULL;
	if (indice) delete[] indice; indice = NULL;
	if (h_sph) delete[] h_sph; h_sph = NULL;
	if (h_poly) delete[] h_poly; h_poly = NULL;
	if (h_mass) delete h_mass; h_mass = NULL;
	if (d_mass) checkCudaErrors(cudaFree(h_mass)); h_mass = NULL;
	if (d_sph) checkCudaErrors(cudaFree(d_sph)); d_sph = NULL;
	if (d_poly) checkCudaErrors(cudaFree(d_poly)); d_poly = NULL;
}

bool polygonObject::define(tImport tm, QTextStream& qs)
{
	switch (tm)
	{
	case MILKSHAPE_3D_ASCII:
		fromMS3DASCII(qs);
		break;
	}

	return true;
}

void polygonObject::fromMS3DASCII(QTextStream& qs)
{
	QString ch;
	//unsigned int nvertex = 0;
	//unsigned int npoly = 0;
	qs >> ch;
	unsigned int ui;
	int begin = ch.indexOf("\"");
	int end = ch.lastIndexOf("\"");
	QString _name = ch.mid(begin + 1, end - 1);
	qs >> ch >> ch >> nvertex;
	vertice = new VEC3D[nvertex];
	double x_max = FLT_MIN; double x_min = FLT_MAX;
	double y_max = FLT_MIN; double y_min = FLT_MAX;
	double z_max = FLT_MIN; double z_min = FLT_MAX;
	for (unsigned int i = 0; i < nvertex; i++){
		qs >> ch >> vertice[i].x >> vertice[i].y >> vertice[i].z >> ch >> ch >> ch;
		vertice[i].x *= 0.001;
		vertice[i].y *= 0.001;
		vertice[i].z *= 0.001;
		if (x_min > vertice[i].x) x_min = vertice[i].x;
		if (x_max < vertice[i].x) x_max = vertice[i].x;
		if (y_min > vertice[i].y) y_min = vertice[i].y;
		if (y_max < vertice[i].y) y_max = vertice[i].y;
		if (z_min > vertice[i].z) z_min = vertice[i].z;
		if (z_max < vertice[i].z) z_max = vertice[i].z;
	}
	org = VEC3D(0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (z_min + z_max));
	qs >> ui;
	for (unsigned int i = 0; i < ui; i++)
		qs >> ch >> ch >> ch;
	qs >> nindex;
	indice = new VEC3UI[nindex];
	for (unsigned int i = 0; i < nindex; i++){
		qs >> ch >> indice[i].x >> indice[i].y >> indice[i].z >> ch >> ch >> ch >> ch;
	}
	h_poly = new host_polygon_info[nindex];
	//h_loc_poly = new host_polygon_info[nindex];
	h_sph = new VEC4D[nindex];
	//h_loc_sph = new VEC4D[nindex];
	double fc = 0;
	double ft = 0.7;
	for (unsigned int i = 0; i < nindex; i++){
		fc = 0;
		host_polygon_info po;
		po.P = vertice[indice[i].x];
		po.Q = vertice[indice[i].y];
		po.R = vertice[indice[i].z];
		po.V = po.Q - po.P;
		po.W = po.R - po.P;
		po.N = po.V.cross(po.W);
		po.N = po.N / po.N.length();
		h_poly[i] = po;
		VEC3D M1 = (po.Q + po.P) / 2;
		VEC3D M2 = (po.R + po.P) / 2;
		VEC3D D1 = po.N.cross(po.V);
		VEC3D D2 = po.N.cross(po.W);
		double t = 0;//t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
		if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
		{
			t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
		}
		else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
		{
			t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
		}
		else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
		{
			t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
		}
		VEC3D Ctri = M1 + t * D1;
		VEC4D sph;
		sph.w = (Ctri - po.P).length();
		sph.x = Ctri.x; sph.y = Ctri.y; sph.z = Ctri.z;
// 		while (abs(fc - ft) > 0.00001)
// 		{
// 			d = ft * sph.w;
// 			double p = d / po.N.length();
// 			VEC3D _c = Ctri - p * po.N;
// 			sph.x = _c.x; sph.y = _c.y; sph.z = _c.z;
// 			sph.w = (_c - po.P).length();
// 			fc = d / sph.w;
// 		}
		if (sph.w > maxRadii)
			maxRadii = sph.w;
		h_sph[i] = sph;
	}
	for (unsigned int i = 0; i < nvertex; i++){
		vertice[i] = vertice[i] - org;
	}
	h_mass = new host_polygon_mass_info;
	h_mass->origin = org;
	h_mass->omega = 0;
	h_mass->vel = 0;
	h_mass->ep = EPD(1.0, 0, 0, 0);
}

unsigned int polygonObject::makeParticles(double rad, VEC3UI &_size, VEC3D& spacing, unsigned int nstack, bool isOnlyCount, VEC4D_PTR pos /* = NULL */, unsigned int sid)
{
	return 0;
}

void polygonObject::save_object_data(QTextStream& ts)
{
	bool isExistMass = ms ? true : false;
	ts << "OBJECT POLYGON " << id << " " << name << " " << roll_type << " " << mat_type << " " << (int)_expression << " " << isExistMass << endl
		<< filePath << endl;

	if (isExistMass)
	{
		save_mass_data(ts);
	}
}

void polygonObject::cuAllocData(unsigned int _np)
{
	if (!d_poly)
		checkCudaErrors(cudaMalloc((void**)&d_poly, sizeof(device_polygon_info) * nindex));
	if (!d_sph)
		checkCudaErrors(cudaMalloc((void**)&d_sph, sizeof(double4) * nindex));
	if (!d_mass)
		checkCudaErrors(cudaMalloc((void**)&d_mass, sizeof(device_polygon_mass_info)));

	updateDeviceFromHost();
}

void polygonObject::updateDeviceFromHost()
{
	checkCudaErrors(cudaMemcpy(d_poly, h_poly, sizeof(device_polygon_info) * nindex, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sph, h_sph, sizeof(double) * 4 * nindex, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mass, h_mass, sizeof(device_polygon_mass_info), cudaMemcpyHostToDevice));
}

void polygonObject::updateFromMass()
{
	mass* m = object::pointMass();
	VEC3D vel = m->getVelocity();
	EPD ep = m->getEP();
	EPD ev = m->getEV();
	org = m->getPosition();
	VEC3D *g_vertice = new VEC3D[nvertex];
	for(unsigned int i = 0; i < nvertex; i++){
		g_vertice[i] = org + m->toGlobal(vertice[i]);
	}
	for (unsigned int i = 0; i < nindex; i++){
		//fc = 0;
		//VEC3D psph = org + m->toGlobal(VEC3D(h_sph[i].x, h_sph[i].y, h_sph[i].z));
		host_polygon_info po;
		po.P = g_vertice[indice[i].x];
		po.Q = g_vertice[indice[i].y];
		po.R = g_vertice[indice[i].z];
		po.V = po.Q - po.P;
		po.W = po.R - po.P;
		po.N = po.V.cross(po.W);
		po.N = po.N / po.N.length();
		h_poly[i] = po;
		VEC3D M1 = (po.Q + po.P) / 2;
		VEC3D M2 = (po.R + po.P) / 2;
		VEC3D D1 = po.N.cross(po.V);
		VEC3D D2 = po.N.cross(po.W);
		double t = 0;//t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
		if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
		{
			t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
		}
		else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
		{
			t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
		}
		else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
		{
			t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
		}
		VEC3D Ctri = M1 + t * D1;
		VEC4D sph;
		sph.w = (Ctri - po.P).length();
		sph.x = Ctri.x; sph.y = Ctri.y; sph.z = Ctri.z;
		h_sph[i] = sph;
		//h_sph[i] = VEC4D(psph.x, psph.y, psph.z, h_sph[i].w);
	}
	h_mass->origin = org;
	h_mass->vel = vel;
	h_mass->omega = VEC3D(
		2.0*(-ep.e1 * ev.e0 + ep.e0 * ev.e1 - ep.e3 * ev.e2 + ep.e2 * ev.e3),
		2.0*(-ep.e2 * ev.e0 + ep.e3 * ev.e1 + ep.e0 * ev.e2 - ep.e1 * ev.e3),
		2.0*(-ep.e3 * ev.e0 - ep.e2 * ev.e1 + ep.e1 * ev.e2 + ep.e0 * ev.e3));
	h_mass->ep = ep;
	updateDeviceFromHost();
	delete[] g_vertice; g_vertice = NULL;
}