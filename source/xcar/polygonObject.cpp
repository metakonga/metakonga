#include "polygonObject.h"
#include "mphysics_cuda_dec.cuh"
#include "numeric_utility.h"
#include <QFile>
#include <QTextStream>

unsigned int polygonObject::nPolygonObject = 0;

polygonObject::polygonObject()
	: pointMass()
	, vertexList(NULL)
	, indexList(NULL)
	, ntriangle(0)
	, maxRadii(0)
	, filePath("")
{

}

polygonObject::polygonObject(QString _name, geometry_use _roll)
	: pointMass(_name, POLYGON_SHAPE, _roll)
	, vertexList(NULL)
	, indexList(NULL)
	, maxRadii(0)
	, ntriangle(0)
	, filePath("")
{
}

polygonObject::polygonObject(const polygonObject& _poly)
	: pointMass(_poly)
	, vertexList(NULL)
	, indexList(NULL)
	, ntriangle(_poly.NumTriangle())
	, maxRadii(_poly.maxRadius())
	, filePath(_poly.meshDataFile())
{
}

polygonObject::~polygonObject()
{
	nPolygonObject--;
}

bool polygonObject::define(import_shape_type t, VEC3D& loc, int _ntriangle, double* vList, unsigned int *iList)
{
	this->setPosition(loc);
	switch (t)
	{
	case MILKSHAPE_3D_ASCII: _fromMS3DASCII(_ntriangle, vList, iList); break;
	case STL_ASCII: _fromSTLASCII(_ntriangle, vList, loc); break;
	}
	return true;
}

void polygonObject::_fromMS3DASCII(int _ntriangle, double* vList, unsigned int *iList)
{
	ntriangle = _ntriangle;
	vertexList = vList;
	indexList = iList;
	unsigned int a, b, c;
	VEC3D P, Q, R, V, W, N;
	VEC3D com;
	double vol = 0.0;
	for (unsigned int i = 0; i < ntriangle; i++)
	{
		a = iList[i * 3 + 0];
		b = iList[i * 3 + 1];
		c = iList[i * 3 + 2];
		P = VEC3D(vList[a * 3 + 0], vList[a * 3 + 1], vList[a * 3 + 2]);
		Q = VEC3D(vList[b * 3 + 0], vList[b * 3 + 1], vList[b * 3 + 2]);
		R = VEC3D(vList[c * 3 + 0], vList[c * 3 + 1], vList[c * 3 + 2]);
		//vol += abs(SignedVolumeOfTriangle(P, Q, R));
		V = Q - P;
		W = R - P;
		N = V.cross(W);
		N = N / N.length();
		VEC3D M1 = (Q + P) / 2;
		VEC3D M2 = (R + P) / 2;
		VEC3D D1 = N.cross(V);
		VEC3D D2 = N.cross(W);
		// 		if (N.z < 0.0)
		// 		{
		// 			bool p = true;
		// 		}
		double t = 0;
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
		double rad = (Ctri - P).length();
		com += Ctri;
		// 		while (abs(fc - ft) > 0.00001)
		// 		{
		// 			d = ft * sph.w;
		// 			double p = d / po.N.length();
		// 			VEC3D _c = Ctri - p * po.N;
		// 			sph.x = _c.x; sph.y = _c.y; sph.z = _c.z;
		// 			sph.w = (_c - po.P).length();
		// 			fc = d / sph.w;
		// 		}
		if (rad > maxRadii)
			maxRadii = rad;
	}
	pointMass::pos = com / ntriangle;
	nPolygonObject++;
}

void polygonObject::_fromSTLASCII(int _ntriangle, double* vList, VEC3D& loc)
{
	ntriangle = _ntriangle;
	vertexList = vList;
// 	unsigned int a, b, c;
	VEC3D P, Q, R;
// 	//VEC3D com;
// 	double _vol = 0.0;
// 	double ixx = 0.0;
// 	double iyy = 0.0;
// 	double izz = 0.0;
// 	double ixy = 0.0;
// 	double ixz = 0.0;
// 	double iyz = 0.0;
	for (unsigned int i = 0; i < ntriangle; i++)
	{
		P = VEC3D(vertexList[i * 9 + 0], vertexList[i * 9 + 1], vertexList[i * 9 + 2]);
		Q = VEC3D(vertexList[i * 9 + 3], vertexList[i * 9 + 4], vertexList[i * 9 + 5]);
		R = VEC3D(vertexList[i * 9 + 6], vertexList[i * 9 + 7], vertexList[i * 9 + 8]);
// 		_vol += numeric::utility::signed_volume_of_triangle(P, Q, R);
// 		VEC3D ctri = numeric::utility::calculate_center_of_triangle(P, Q, R);
// 		ctri = ctri - loc;
		//com += ctri;
// 		ixx += ctri.y * ctri.y + ctri.z * ctri.z;
// 		iyy += ctri.x * ctri.x + ctri.z * ctri.z;
// 		izz += ctri.x * ctri.x + ctri.y * ctri.y;
// 		ixy -= ctri.x * ctri.y;
// 		ixz -= ctri.x * ctri.z;
// 		iyz -= ctri.y * ctri.z;
		P = this->toLocal(P - pos);
		Q = this->toLocal(Q - pos);
		R = this->toLocal(R - pos);
		vertexList[i * 9 + 0] = P.x;
		vertexList[i * 9 + 1] = P.y;
		vertexList[i * 9 + 2] = P.z;
		vertexList[i * 9 + 3] = Q.x;
		vertexList[i * 9 + 4] = Q.y;
		vertexList[i * 9 + 5] = Q.z;
		vertexList[i * 9 + 6] = R.x;
		vertexList[i * 9 + 7] = R.y;
		vertexList[i * 9 + 8] = R.z;
	}
// 	object::vol = _vol;
// 	object::dia_iner0 = VEC3D(ixx, iyy, izz) / ntriangle;
// 	object::sym_iner0 = VEC3D(ixy, ixz, iyz) / ntriangle;
// 	pointMass::ms = vol * d;
// 	pointMass::diag_iner = pointMass::ms * object::dia_iner0;
// 	pointMass::sym_iner = pointMass::ms * object::sym_iner0;
	nPolygonObject++;
}

//void polygonObject::cuAllocData(unsigned int _np)
//{
// 	if (!d_poly)
// 		checkCudaErrors(cudaMalloc((void**)&d_poly, sizeof(device_polygon_info) * nindex));
// 	if (!d_sph)
// 		checkCudaErrors(cudaMalloc((void**)&d_sph, sizeof(double4) * nindex));
// 	if (!d_mass)
// 		checkCudaErrors(cudaMalloc((void**)&d_mass, sizeof(device_polygon_mass_info)));
// 
// 	updateDeviceFromHost();
//}

void polygonObject::updateDeviceFromHost()
{
	//checkCudaErrors(cudaMemcpy(d_poly, h_poly, sizeof(device_polygon_info) * nindex, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_sph, h_sph, sizeof(double) * 4 * nindex, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_mass, h_mass, sizeof(device_polygon_mass_info), cudaMemcpyHostToDevice));
}

// void polygonObject::update(pointMass* pm)
// {
//  	mass* m = object::pointMass();
//  	VEC3D vel = m->getVelocity();
//  	EPD ep = m->getEP();
//  	EPD ev = m->getEV();
//  	org = m->Position();
//  	VEC3D *g_vertice = new VEC3D[nvertex];
//  	for(unsigned int i = 0; i < nvertex; i++){
//  		g_vertice[i] = org + m->toGlobal(vertice[i]);
//  	}
//  	for (unsigned int i = 0; i < nindex; i++){
//  		//fc = 0;
//  		//VEC3D psph = org + m->toGlobal(VEC3D(h_sph[i].x, h_sph[i].y, h_sph[i].z));
//  		host_polygon_info po;
//  		po.P = g_vertice[indice[i].x];
//  		po.Q = g_vertice[indice[i].y];
//  		po.R = g_vertice[indice[i].z];
//  		po.V = po.Q - po.P;
//  		po.W = po.R - po.P;
//  		po.N = po.V.cross(po.W);
//  		po.N = po.N / po.N.length();
//  		h_poly[i] = po;
//  		VEC3D M1 = (po.Q + po.P) / 2;
//  		VEC3D M2 = (po.R + po.P) / 2;
//  		VEC3D D1 = po.N.cross(po.V);
//  		VEC3D D2 = po.N.cross(po.W);
//  		double t = 0;//t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
//  		if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
//  		{
//  			t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
//  		}
//  		else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
//  		{
//  			t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
//  		}
//  		else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
//  		{
//  			t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
//  		}
//  		VEC3D Ctri = M1 + t * D1;
//  		VEC4D sph;
//  		sph.w = (Ctri - po.P).length();
//  		sph.x = Ctri.x; sph.y = Ctri.y; sph.z = Ctri.z;
//  		h_sph[i] = sph;
//  		//h_sph[i] = VEC4D(psph.x, psph.y, psph.z, h_sph[i].w);
//  	}
//  	h_mass->origin = org;
//  	h_mass->vel = vel;
//  	h_mass->omega = VEC3D(
//  		2.0*(-ep.e1 * ev.e0 + ep.e0 * ev.e1 - ep.e3 * ev.e2 + ep.e2 * ev.e3),
//  		2.0*(-ep.e2 * ev.e0 + ep.e3 * ev.e1 + ep.e0 * ev.e2 - ep.e1 * ev.e3),
//  		2.0*(-ep.e3 * ev.e0 - ep.e2 * ev.e1 + ep.e1 * ev.e2 + ep.e0 * ev.e3));
//  	h_mass->ep = ep;
//  	updateDeviceFromHost();
//  	delete[] g_vertice; g_vertice = NULL;
// }