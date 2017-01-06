#include "parSIM.h"
#include "cublas_v2.h"
#include "cu_dem_dec.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

using namespace parSIM;
using namespace parSIM::geo;

geometry::geometry(Simulation *_sim, std::string& _name)
	: sim(_sim)
	, name(_name)
	, isParticleGeometry(false)
	, pm(NULL)
{
	
}

geometry::~geometry()
{

}

void geometry::setMaterial()
{
	switch(mat_type)
	{
	case ACRYLIC:
		material.density = static_cast<double>(ACRYLIC_DENSITY);
		material.youngs = static_cast<double>(ACRYLIC_YOUNG_MODULUS);
		material.poisson = static_cast<double>(ACRYLIC_POISSON_RATIO);
		break;
	case STEEL:
		material.density = static_cast<double>(STEEL_DENSITY);
		material.youngs = static_cast<double>(STEEL_YOUNGS_MODULUS);
		material.poisson = static_cast<double>(STEEL_POISSON_RATIO);
		break;
	}
}

plane::plane(Simulation* _sim, std::string _name)
	: geometry(_sim, _name)
	, l1(0)
	, l2(0)
{
	geometry::geo_type = PLANE;
}

plane::plane(const plane& _plane)
	: geometry(_plane.get_sim(), _plane.get_name())
	, l1(_plane.l1)
	, l2(_plane.l2)
{
	size = _plane.size;
	pa = _plane.pa;
	pb = _plane.pb;
	u1 = _plane.u1;
	u2 = _plane.u2;
	uw = _plane.uw;
	xw = _plane.xw;
}

plane::~plane()
{

}

bool plane::setSpecificDataFromFile(std::fstream& pf)
{
	return true;
}

void plane::insert2simulation()
{
	if(sim)
	{
		sim->insert_geometry(name, this);
	}
}

bool plane::define_geometry()
{
	setMaterial();
	pa -= xw;
	pb -= xw;
	l1 = pa.length();
	l2 = pb.length();
	u1 = pa / l1;
	u2 = pb / l2;
	uw = u1.cross(u2);
	return true;
}

bool plane::define_device_info()
{
	return true;
}

bool plane::save2file(std::fstream& of)
{
	int type = (int)PLANE;
	of.write((char*)&type, sizeof(int));
	save_plane_info spi = {xw.x, xw.y, xw.z, pa.x, pa.y, pa.z, pb.x, pb.y, pb.z};
	of.write((char*)&spi, sizeof(spi));
	return true;
}

void plane::cu_hertzian_contact_force(contact_coefficient& coe, 
									  bool* isLineContact,
									 double* pos,
									 double* vel, 
									 double* omega,
									 double* force,
									 double* moment, 
									 unsigned int np,
									 unsigned int* u1,
									 unsigned int* u2,
									 unsigned int* u3)
{
	//cu_cube_hertzian_contact_force(d_planes, coe.kn, coe.vn, coe.ks, coe.vs, coe.mu, pos, vel, omega, force, moment, np);
}


cube::cube(Simulation *_sim, std::string _name)
	: geometry(_sim, _name)
{

}

cube::cube(const cube& _cube)
	: geometry(_cube.get_sim(), _cube.get_name())
{
	size = _cube.size;
	//memcpy(cube_plane, _cube.cube_plane, sizeof(plane)*6);
	planes = planes;
}

cube::~cube()
{
	for(std::map<std::string,plane*>::iterator p = planes.begin(); p != planes.end(); p++){
		delete p->second;
	}
}

void cube::define(vector3<double> _size, vector3<double> pos, material_type mtype, geometry_use guse, geometry_type gtype)
{
	size = _size;
	geometry::position = pos;
	geometry::geo_use = guse;
	geometry::geo_type = gtype;
	geometry::mat_type = mtype;
	sim->insert_geometry(name, this);
}

void cube::insert2simulation()
{
	if(sim) 
		sim->insert_geometry(name, this);
}

bool cube::setSpecificDataFromFile(std::fstream& pf)
{
	return true;
}

bool cube::define_geometry()
{
	std::cout << "    Define geometry of cube - " << name << " ";
	setMaterial();
	//sim->add_pair_material_condition(this->mat_type);
	plane* bottom = new plane(NULL, "bottom");
	bottom->size = vector2<double>(size.x, size.z);
	bottom->xw = position;
	bottom->pa = position + vector3<double>(0, 0, size.z);
	bottom->pb = position + vector3<double>(size.x, 0, 0);
	bottom->define_geometry();
	planes["bottom"] = bottom;
	std::cout << ".";
	plane* left = new plane(NULL, "left");
	left->size = vector2<double>(size.y, size.z);
	left->xw = position;
	left->pa = position + vector3<double>(0, size.y, 0);
	left->pb = position + vector3<double>(0, 0, size.z);
	left->define_geometry();
	planes["left"] = left;
	std::cout << ".";
	plane* right = new plane(NULL, "right");
	right->size = vector2<double>(size.y, size.z);
	right->xw = position + vector3<double>(size.x, 0, 0);
	right->pa = position + vector3<double>(size.x, 0, size.z);
	right->pb = position + vector3<double>(size.x, size.y, 0);
	right->define_geometry();
	planes["right"] = right;
	std::cout << ".";
	plane* back = new plane(NULL, "back");
	back->size = vector2<double>(size.x, size.y);
	back->xw = position;
	back->pa = position + vector3<double>(size.x, 0, 0);
	back->pb = position + vector3<double>(0, size.y, 0);
	back->define_geometry();
	planes["back"] = back;
	std::cout << ".";
	plane* front = new plane(NULL, "front");
	front->size = vector2<double>(size.x, size.y);
	front->xw = position + vector3<double>(0, 0, size.z);
	front->pa = position + vector3<double>(0, size.y, size.z);
	front->pb = position + vector3<double>(size.x, 0, size.z);
	front->define_geometry();
	planes["front"] = front;
	std::cout << ".";
	plane* top = new plane(NULL, "top");
	top->size = vector2<double>(size.x, size.z);
	top->xw = position + vector3<double>(0, size.y, 0);
	top->pa = position + vector3<double>(size.x, size.y, 0);
	top->pb = position + vector3<double>(0, size.y, size.z);
	top->define_geometry();
	planes["top"] = top;
	std::cout << ". Ok" << std::endl;
	return true;
}

void cube::hertzian_contact_force(double r, double dt, contact_coefficient& coe, vector3<double>& pos, vector3<double>& vel, vector3<double>& omega, vector3<double>& nforce, vector3<double>& sforce)
{
	vector3<double> single_force;
	for(std::map<std::string, plane*>::iterator _p = planes.begin(); _p != planes.end(); _p++){
		plane* p = _p->second; 
		vector3<double> dp = pos - p->xw;
		vector3<double> wp = vector3<double>(dp.dot(p->u1), dp.dot(p->u2), dp.dot(p->uw));
		if(abs(wp.z) < r && (wp.x > 0 && wp.x < p->l1) && (wp.y > 0 && wp.y < p->l2)){
			vector3<double> uu = p->uw / p->uw.length();
			double pp = -sign(dp.dot(p->uw));
			vector3<double> unit = pp * uu;
			/*vector3<double> unit = -sign((pos - p->xw).dot(p->uw)) * (p->uw / p->uw.length());*/
			double collid_dist = r - abs(dp.dot(unit));
			vector3<double> dv = -(vel + omega.cross(r * unit));
			single_force = (-coe.kn * pow(collid_dist, 1.5) + coe.vn * dv.dot(unit)) * unit;
			vector3<double> e = dv - dv.dot(unit) * unit;
			double mag_e = e.length();
			if(mag_e){
				vector3<double> s_hat = e / mag_e;
				double ds = mag_e * dt;
				vector3<double> shear_force = std::min(coe.ks * ds + coe.vs * dv.dot(s_hat), coe.mu * single_force.length()) * s_hat;
				sforce += (r * unit).cross(shear_force);
			}
			nforce += single_force;
		}
	}
}

void cube::cu_hertzian_contact_force(contact_coefficient& coe, 
									 bool* isLineContact,
									 double* pos,
									 double* vel, 
									 double* omega,
									 double* force,
									 double* moment, 
									 unsigned int np,
									 unsigned int* u1,
									 unsigned int* u2,
									 unsigned int* u3)
{
	cu_cube_hertzian_contact_force(d_planes, coe.kn, coe.vn, coe.ks, coe.vs, coe.mu, isLineContact, pos, vel, omega, force, moment, np);
}

bool cube::save2file(std::fstream& of)
{
	int type = CUBE;
	unsigned int fsize = sizeof(double);
	of.write((char*)&fsize, sizeof(unsigned int));
	of.write((char*)&type, sizeof(int));
	int name_size = name.size();
	of.write((char*)&name_size, sizeof(int));
	of.write((char*)name.c_str(), sizeof(char) * name_size);
	vector3<double> vertex[8] = {0, };
	save_cube_info sci = {position.x, position.y, position.z, size.x, size.y, size.z};
	vertex[0] = position;
	vertex[1] = position + vector3<double>(0, size.y, 0);
	vertex[2] = position + vector3<double>(0, 0, size.z);
	vertex[3] = position + vector3<double>(0, size.y, size.z);
	vertex[4] = position + vector3<double>(size.x, 0, size.z);
	vertex[5] = position + vector3<double>(size.x, size.y, size.z);
	vertex[6] = position + vector3<double>(size.x, 0, 0);
	vertex[7] = position + vector3<double>(size.x, size.y, 0);

	of.write((char*)&sci, sizeof(save_cube_info));
	of.write((char*)vertex, sizeof(double)*3 * 8);
	
	return true;
}

bool cube::define_device_info()
{
	device_plane_info *dpi = new device_plane_info[6];
	int i = 0;
	for(std::map<std::string, plane*>::iterator p = planes.begin(); p != planes.end(); p++, i++)
	{
		dpi[i].l1 = p->second->l1;
		dpi[i].l2 = p->second->l2;
		dpi[i].xw = make_double3(p->second->xw.x, p->second->xw.y, p->second->xw.z);
		dpi[i].uw = make_double3(p->second->uw.x, p->second->uw.y, p->second->uw.z);
		dpi[i].u1 = make_double3(p->second->u1.x, p->second->u1.y, p->second->u1.z);
		dpi[i].u2 = make_double3(p->second->u2.x, p->second->u2.y, p->second->u2.z);
		dpi[i].pa = make_double3(p->second->pa.x, p->second->pa.y, p->second->pa.z);
		dpi[i].pb = make_double3(p->second->pb.x, p->second->pb.y, p->second->pb.z);
	}
	checkCudaErrors( cudaMalloc((void**)&d_planes, sizeof(device_plane_info) * 6) );
	checkCudaErrors( cudaMemcpy(d_planes, dpi, sizeof(device_plane_info) * 6, cudaMemcpyHostToDevice) );

	delete [] dpi; 
	dpi = NULL;

	return true;
}

int shape::id = 0;

shape::shape(Simulation* _sim, std::string _name)
	: geometry(_sim, _name)
	, id_set(NULL)
	, poly_start(NULL)
	, poly_end(NULL)
	, d_polygons(NULL)
	, d_id_set(NULL)
	, d_poly_start(NULL)
	, d_poly_end(NULL)
	, d_vertice(NULL)
	, d_local_vertice(NULL)
	, d_mass_info(NULL)
	, d_indice(NULL)
	, d_body_force(NULL)
	, isLineContact(false)
	, isUpdate(false)
{

}

shape::~shape()
{
	if(id_set) delete [] id_set; id_set = NULL;
	if(poly_start) delete [] poly_start; poly_start = NULL;
	if(poly_end) delete [] poly_end; poly_end = NULL;

	if(d_vertice) checkCudaErrors( cudaFree(d_vertice) );
	if(d_body_force) checkCudaErrors( cudaFree(d_body_force) );
	if(d_local_vertice) checkCudaErrors( cudaFree(d_local_vertice) );
	if(d_polygons) checkCudaErrors( cudaFree(d_polygons) );
	if(d_id_set) checkCudaErrors( cudaFree(d_id_set) );
	if(d_indice) checkCudaErrors( cudaFree(d_indice) );
	if(d_poly_start) checkCudaErrors( cudaFree(d_poly_start) );
	if(d_poly_end) checkCudaErrors( cudaFree(d_poly_end) );
	if(d_mass_info) checkCudaErrors( cudaFree(d_mass_info) );
}

void shape::define(std::string fpath, vector3<double> pos, material_type mtype, geometry_use guse, geometry_type gtype)
{
	file_path = fpath;
	geometry::position = pos;
	geometry::mat_type = mtype;
	geometry::geo_use = guse;
	geometry::geo_type = gtype;
	define_geometry();
	sim->insert_geometry(name, this);
}

void shape::insert2simulation()
{
	if(sim) 
		sim->insert_geometry(name, this);
}

bool shape::setSpecificDataFromFile(std::fstream& pf)
{
	float tv3[3] = {0, };
	pf.read((char*)tv3, sizeof(float)*3);
	pf.read((char*)tv3, sizeof(float)*3);

	position.x = static_cast<double>(tv3[0]);
	position.y = static_cast<double>(tv3[1]);
	position.z = static_cast<double>(tv3[2]);
	//position.z = 0.25;
	vector3<double> velocity(static_cast<double>(tv3[0]),
		static_cast<double>(tv3[1]),
		static_cast<double>(tv3[2])
	);
	if(pm){
		pm->Position() = position;
		pm->Velocity() = 0.0;//velocity;
		update_polygons();
	}
	return true;
}

void shape::Rotation()
{
	for(unsigned int i = 0; i < vertice.sizes(); i++){
		vector3<double> vertex = position + pm->TransformationMatrix() * l_vertice(i);
		vertice(i) = vertex;
	}
}

bool shape::define_geometry()
{
	std::stringstream ss;
	std::cout << "    Define geometry of shape - " << name << " ";
	setMaterial();
	std::string str;
	std::fstream pf;
	vector3<double> vertex;
	vector3<double> l_vertex;
	//vector<vector3<double>> vertice;
	vector3<unsigned int> index;

	//particles *ps = sim->getParticles();
	//vector3<double> com(61.048 / 1000, 40.1/1000, 7/1000);
	//vector3<double> com(60.684 / 1000, 5.015/1000, -4.546/1000);
	vector3<double> com(0.0 / 1000, 0.0/1000, 0.0/1000);
	pf.open(file_path, std::ios::in);

	double max_height = 0;
	while(!pf.eof()){
		pf >> str;
		if(str=="GRID*")
		{
			pf >> str >> vertex.x >> vertex.y >> str >> str >> str >> vertex.z;
			vertex = vertex / 1000.0;
			l_vertex = vertex - com;
			vertex = l_vertex + position;
			vertice.push(vertex);
			l_vertice.push(l_vertex);
		}
		if(str=="CTRIA3"){
			pf >> str >> str >> index.x >> index.y >> index.z;
			indice.push(index - vector3<unsigned int>(1,1,1));
		}
	}
	
	vertice.adjustment();
	indice.adjustment();
	std::cout << ".";
	//ps->resize_pos(ps->Size() + vertice.sizes());
	//
	//spos = ps->add_pos_v3data(vertice.get_ptr(), -1, vertice.sizes());

	polygon po;
	std::fstream pf2;
	//pf2.open("C:/C++/result/ttttttt.txt", std::ios::out);
	double t = 0;
	for(vector3<unsigned int>* id = indice.begin(); id != indice.end(); id++)
	{
		po.P = vertice(id->x);
		//pf2 << po.P.x << " " << po.P.y << " " << po.P.z << "---- po.P" << std::endl;
		po.Q = vertice(id->y);
		//pf2 << po.Q.x << " " << po.Q.y << " " << po.Q.z << "---- po.Q" << std::endl;
		po.R = vertice(id->z);
		//pf2 << po.R.x << " " << po.R.y << " " << po.R.z << "---- po.R" << std::endl;
		po.V = po.Q - po.P;
		double length = po.V.length();
		po.W = po.R - po.P;
		length = po.W.length();
		po.N = po.V.cross(po.W);
		//pf2 << po.N.x << " " << po.N.y << " " << po.N.z << "---- po.N" << std::endl;
		polygons.push(po);
		/*if(po.P.y < 0.234162 && po.Q.y < 0.234162 && po.R.y < 0.234162){
			if(po.N.y < 0)
			{
				bool pause = true;
			}
		}*/
	}
	//pf2.close();
	polygons.adjustment();
	std::cout << ".";
	id_set = new unsigned[polygons.sizes() * 3];
	unsigned *point_set = new unsigned[polygons.sizes() * 3];
	unsigned int cnt = 0;
	unsigned int id_cnt = 0;
	for(vector3<unsigned int>* id = indice.begin(); id != indice.end(); id++, cnt += 3, id_cnt++){
		id_set[cnt]   = id_cnt; point_set[cnt]   = id->x;
		id_set[cnt+1] = id_cnt; point_set[cnt+1] = id->y;
		id_set[cnt+2] = id_cnt; point_set[cnt+2] = id->z;
	}
	std::cout << ".";
	thrust::sort_by_key(point_set, point_set + polygons.sizes() * 3, id_set);
	unsigned int v_size = vertice.sizes();
	poly_start = new unsigned int[v_size];
	poly_end = new unsigned int[v_size];
	cnt = 0;
	for(unsigned int i = 0; i < v_size; i++){
		unsigned int p_id = i;
		poly_start[p_id] = cnt;
		while( point_set[cnt] == p_id )
		{
			/*std::cout << point_set[cnt - 1] << std::endl;*/
			cnt++;
		}

		poly_end[p_id] = cnt;
	}
	std::cout << ".";
	delete [] point_set;
	std::cout << " Ok" << std::endl;
	return true;
}

void shape::update_polygons()
{
	
	vector4<double> *spos = new vector4<double>[vertice.sizes()];

	vector3<double> vertex;
	matrix3x3<double> A = pm->TransformationMatrix();
	for(unsigned int i = 0; i < vertice.sizes(); i++){
		vertex = position + A * l_vertice(i);
		spos[i] = vector4<double>(vertex.x, vertex.y, vertex.z, spos[i].w);
	}

	polygon po;
	unsigned int i = 0;
	
	for(vector3<unsigned int>* id = indice.begin(); id != indice.end(); id++, i++)
	{
		po.P = vector3<double>(spos[id->x].x, spos[id->x].y, spos[id->x].z);
		po.Q = vector3<double>(spos[id->y].x, spos[id->y].y, spos[id->y].z);
		po.R = vector3<double>(spos[id->z].x, spos[id->z].y, spos[id->z].z);
		po.V = po.Q - po.P;
		//double length = po.V.length();
		po.W = po.R - po.P;
		//length = po.W.length();
		po.N = po.V.cross(po.W);
		polygons(i) = po;
	}
	isUpdate = true;
	delete [] spos;
}

vector3<double> shape::ClosestPtPointTriangle(
	vector3<double>& p, 
	vector3<double>& a, 
	vector3<double>& b, 
	vector3<double>& c, 
	with_contact *wc)
{
	// Check if P in vertex region outside A
	vector3<double> ab = b - a;
	vector3<double> ac = c - a;
	vector3<double> ap = p - a;

	double d1 = ab.dot(ap);
	double d2 = ac.dot(ap);
	if(d1 <= 0.0 && d2 <= 0.0){
		*wc = VERTEX;
		return a; // barycentric coordinates (1, 0, 0);
	}

	// Check if P in vertex region outside B
	vector3<double> bp = p - b;
	double d3 = ab.dot(bp);
	double d4 = ac.dot(bp);
	if(d3 >= 0.0 && d4 <= d3) {
		*wc = VERTEX;
		return b; // barycentric coordinates (0, 1, 0)
	}

	// Check if P in edge region of AB, if so return projection of P onto AB
	double vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0 && d3 <= 0.0){
		*wc = LINE;
		double v = d1 / (d1 - d3);
		return a + v * ab; // barycentric coordinates (1-v, v, 0)
	}

	// Check if P in vertex region outside C
	vector3<double> cp = p - c;
	double d5 = ab.dot(cp);
	double d6 = ac.dot(cp);
	if(d6 >= 0.0 && d5 <= d6) {
		*wc = VERTEX;
		return c; // barycentric coordinates (0, 0, 1)
	}

	// Check if P in edge region of AC, if so return projection of P onto AC
	double vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
		*wc = LINE;
		double w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	double va = d3 * d6 - d5 * d4;
	if(va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
		*wc = LINE;
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	*wc = PLANE;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;

	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

bool shape::hertzian_contact_force(
	unsigned int id,
	double r, 
	contact_coefficient& coe,
	vector3<double>& pos, 
	vector3<double>& vel,
	vector3<double>& omega,
	vector3<double>& force,
	vector3<double>& moment,
	vector3<double>& line_force,
	vector3<double>& line_moment)
{
	contact_count = 0;
	contact_info cinfo[10];
	unsigned int poly_id = 0;
	polygon poly;
	with_contact wc;
	vector3<double> sp, contact_point, unit, single_force;
	double dist, collid_dist;
	double dt = Simulation::dt;
	for(unsigned int i = poly_start[id]; i < poly_end[id]; i++){
		poly_id = id_set[i];
		poly = polygons(poly_id);
		contact_point = ClosestPtPointTriangle(pos, poly.P, poly.Q, poly.R, &wc);

		sp = pos - contact_point;
		dist = sp.length();
		vector3<double> s_unit = -sp / dist;
		unit = -poly.N / poly.N.length();
		if(s_unit.dot(unit) <= 0)
			continue;
		collid_dist = r - dist;
		
		if(collid_dist <= 0) 
			continue;
	
		if(wc == LINE || wc == VERTEX)
			cinfo[contact_count].unit = s_unit;
		else
			cinfo[contact_count].unit = unit;

		cinfo[contact_count].contact_with = wc;
		cinfo[contact_count].contact_point = contact_point;
		cinfo[contact_count++].penetration = collid_dist;
	}

	int select_info_id = 0;
	for(int i = 0; i < contact_count; i++){
		if(cinfo[i].contact_with == PLANE){
			collid_dist = cinfo[i].penetration;
			unit = cinfo[i].unit;
			vector3<double> dv = pm->Velocity() - (vel + omega.cross(r * unit));
			single_force = (-coe.kn * pow(collid_dist, 1.5) + coe.vn * dv.dot(unit)) * unit;
			vector3<double> e = dv - dv.dot(unit) * unit;
			double mag_e = e.length();
			if(mag_e){
				vector3<double> s_hat = e / mag_e;
				double ds = mag_e * dt;
				vector3<double> shear_force = std::min(coe.ks * ds + coe.vs * dv.dot(s_hat), coe.mu * single_force.length()) * s_hat;
				moment += (r * unit).cross(shear_force);
			}
			force += single_force;

			plane_body_force += -single_force;
			body_force += -single_force;

			return true;
		}
	}
	for(int i = 0; i < contact_count; i++){
		if(cinfo[i].contact_with == LINE || cinfo[i].contact_with == VERTEX){
			collid_dist = cinfo[i].penetration;
			unit = cinfo[i].unit;
			vector3<double> dv = pm->Velocity() - (vel + omega.cross(r * unit));
			line_force = (-coe.kn * pow(collid_dist, 1.5) + coe.vn * dv.dot(unit)) * unit;
			vector3<double> e = dv - dv.dot(unit) * unit;
			double mag_e = e.length();
			if(mag_e){
				vector3<double> s_hat = e / mag_e;
				double ds = mag_e * dt;
				vector3<double> shear_force = std::min(coe.ks * ds + coe.vs * dv.dot(s_hat), coe.mu * single_force.length()) * s_hat;
				line_moment = (r * unit).cross(shear_force);
			}
			isLineContact = true;
			line_contact_force = -line_force;
			return false;
		}
	}
	return false;
}

bool shape::save2file(std::fstream& of)
{

	int type = SHAPE;
	of.write((char*)&type, sizeof(int));

	int name_size = name.size();
	of.write((char*)&name_size, sizeof(int));
	of.write((char*)name.c_str(), sizeof(char) * name_size);

	of.write((char*)&vertice.sizes(), sizeof(unsigned int));
	of.write((char*)&indice.sizes(), sizeof(unsigned int));
	of.write((char*)&position, sizeof(vector3<double>));
	of.write((char*)l_vertice.get_ptr(), sizeof(vector3<double>) * vertice.sizes());
	of.write((char*)indice.get_ptr(), sizeof(vector3<unsigned int>) * indice.sizes());

// 	if(pm)
// 	{
// 		switch(subSim->getSolver()){
// 		case MBD:
// 			{
// 				std::map<std::string, pointmass*>::iterator pm = subSim->getMasses()->begin();
// 				if(pm != subSim->getMasses()->end()){
// 					for(; pm != subSim->getMasses()->end(); pm++){
// 						int type = MASS;
// 						of.write((char*)&type, sizeof(int));
// 						//of.write((char*)&time, sizeof(double));
// 						pm->second->save2file(of);
// 					}
// 				}
// 			}
// 			break;
// 		}
// 	}
	//delete [] t_vertice; t_vertice = NULL;
	return true;
}

bool shape::define_device_info()
{
	unsigned int npoly = polygons.sizes();

	unsigned int nmass = sim->getMasses()->size() - 1;
	cu_mass_info mi;
	checkCudaErrors( cudaMalloc((void**)&d_mass_info, sizeof(cu_mass_info)) );
	checkCudaErrors( cudaMalloc((void**)&d_local_vertice, sizeof(double3) * vertice.sizes()) );
	checkCudaErrors( cudaMalloc((void**)&d_vertice, sizeof(double3) * vertice.sizes()) );
	std::map<std::string, pointmass*>::iterator pm = sim->getMasses()->begin(); pm++;
		
	mi.pos = make_double3(pm->second->Position().x, pm->second->Position().y, pm->second->Position().z);
	mi.vel = make_double3(pm->second->Velocity().x, pm->second->Velocity().y, pm->second->Velocity().z);
	
	checkCudaErrors( cudaMemcpy(d_mass_info, &mi, sizeof(cu_mass_info), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_local_vertice, l_vertice.get_ptr(), sizeof(double3) * vertice.sizes(), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_vertice, vertice.get_ptr(), sizeof(double3)* vertice.sizes(), cudaMemcpyHostToDevice) );
	
	checkCudaErrors( cudaMalloc((void**)&d_polygons, sizeof(cu_polygon) * npoly) );
	checkCudaErrors( cudaMalloc((void**)&d_id_set, sizeof(unsigned int) * npoly * 3) );
	checkCudaErrors( cudaMalloc((void**)&d_poly_start, sizeof(unsigned int) * vertice.sizes()) );
	checkCudaErrors( cudaMalloc((void**)&d_poly_end, sizeof(unsigned int) * vertice.sizes()) );
	checkCudaErrors( cudaMalloc((void**)&d_indice, sizeof(uint3) * indice.sizes()) );
	checkCudaErrors( cudaMalloc((void**)&d_body_force, sizeof(double3)) );
// 	checkCudaErrors( cudaMalloc((void**)&d_line_body_force, sizeof(double3)) );
// 	checkCudaErrors( cudaMalloc((void**)&d_body_force, sizeof(double3)) );
// 	checkCudaErrors( cudaMalloc((void**)&d_body_moment, sizeof(double3)) );

// 	checkCudaErrors( cudaMemset(d_line_body_force, 0, sizeof(double3)) );
 	checkCudaErrors( cudaMemset(d_body_force, 0, sizeof(double3)) );
// 	checkCudaErrors( cudaMemset(d_body_moment, 0, sizeof(double3)) );
	//checkCudaErrors( cudaMemset(d_body_force, 0, sizeof(double) * sim->getParticles()->Size() * 3) );
	checkCudaErrors( cudaMemcpy(d_polygons, polygons.get_ptr(), sizeof(cu_polygon) * npoly, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_id_set, id_set, sizeof(unsigned int) * npoly * 3, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_poly_start, poly_start, sizeof(unsigned int) * vertice.sizes(), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_poly_end, poly_end, sizeof(unsigned int) * vertice.sizes(), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_indice, indice.get_ptr(), sizeof(uint3) * indice.sizes(), cudaMemcpyHostToDevice) );
 	return true;
}

void shape::cu_hertzian_contact_force(contact_coefficient& coe,
									  bool* isLineContact,
									  double* pos,
									  double* vel, 
									  double* omega,
									  double* force,
									  double* moment, 
									  unsigned int np,
									  unsigned int *sorted_id,
									  unsigned int *cell_start,
									  unsigned int *cell_end)
{
	double3 bf = cu_shape_hertzian_contact_force(
		d_mass_info,
		d_polygons,
		d_id_set,
		d_poly_start,
		d_poly_end,
		coe.kn,
		coe.vn,
		coe.ks,
		coe.vs,
		coe.mu,
		isLineContact,
		pos,
		vel,
		omega,
		force,
		moment,
		d_body_force,
		np,
		sorted_id,
		cell_start,
		cell_end);
	body_force.x = -bf.x;
	body_force.y = -bf.y;
	body_force.z = -bf.z;
}

void shape::cu_update_geometry(double* A, double3* pos)
{
	//particles *ps = sim->getParticles();
	cu_update_polygons_position(pos, A, d_local_vertice, vertice.sizes(), d_vertice);
	cu_update_polygons_information(d_polygons, d_indice, d_vertice, indice.sizes());
} 