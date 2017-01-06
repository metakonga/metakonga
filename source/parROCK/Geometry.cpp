#include "Geometry.h"
#include "contact.h"
#include "Simulation.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

Geometry::Geometry(Simulation *baseSimulation, std::string geo_name, geometry_type geo_type, geometry_shape geo_shape)
	: sim(baseSimulation)
	, name(geo_name)
	, geoType(geo_type)
	, geoShape(geo_shape)
	, isOnContact(true)
{
	if(sim)
		sim->Geometries()[name] = this;
}

void geo::Line::Define(vector3<double>& sp,  vector3<double>& ep,  vector3<double>& nor)
{
	startPoint = sp;
	endPoint = ep;
	length = (ep - sp).length();
	normal = nor;
}

bool geo::Line::Collision(ball *ib, unsigned id)
{
	if (isOnContact){
		vector3<double> ab = endPoint - startPoint;
		double t = (ib->Position() - startPoint).dot(ab) / ab.dot();
		if (t < 0.0) t = 0.0;
		if (t > 1.0) t = 1.0;
		vector3<double> d = startPoint + t * ab;
		vector3<double> rp = ib->Position() - d;
		double dist = rp.length();
		double cdist = ib->Radius() - dist;
		if (cdist <= 0){
			std::map<Geometry*, ccontact>::iterator it = ib->ContactWMap().find(this);
			if (it != ib->ContactWMap().end()){
				ib->ContactWMap().erase(this);
			}
			return false;
		}
		std::map<Geometry*, ccontact>::iterator it = ib->ContactWMap().find(this);
		if (it != ib->ContactWMap().end()){
			ccontact* c = &(it->second);
			c->CalculateContactForces(cdist, -normal);
		}
		else{
			ccontact c;
			c.SetIBall(ib);
			c.SetJBall(NULL);
			c.SetWall(this);
			c.CalculateContactForces(cdist, -normal);
			ib->InsertWContact(this, c);
		}
	}
	return true;
}

void geo::Line::save2file(std::fstream& of, char ft)
{
	int type = LINE;
	if (ft == 'b'){
		of.write((char*)&type, sizeof(int));
		int name_size = name.size();
		of.write((char*)&name_size, sizeof(int));
		of.write((char*)name.c_str(), sizeof(char) * name_size);
		of.write((char*)&startPoint, sizeof(vector3<double>));
		of.write((char*)&endPoint, sizeof(vector3<double>));
	}
	else if (ft == 'a'){

	}
	else{

	}
}

void geo::Rectangle::Define(vector2<double>& sp, vector2<double>& ep)
{
	startPoint = sp;
	endPoint = ep;
	sizex = ep.x - sp.x;
	sizey = ep.y - sp.y;
	area = sizex * sizey;
}

void geo::Rectangle::save2file(std::fstream& of, char ft)
{
	int type = RECTANGLE;
	vector3<double> lines[4] = { vector3<double>(startPoint.x,         startPoint.y,         0.0), 
								 vector3<double>(startPoint.x + sizex, startPoint.y,         0.0),
								 vector3<double>(startPoint.x,         startPoint.y + sizey, 0.0),
								 vector3<double>(startPoint.x + sizex, startPoint.y + sizey, 0.0) };
	if (ft == 'b'){
		of.write((char*)&type, sizeof(int));
		int name_size = name.size();
		of.write((char*)&name_size, sizeof(int));
		of.write((char*)name.c_str(), sizeof(char) * name_size);
		of.write((char*)&lines[0], sizeof(vector3<double>));
		of.write((char*)&lines[1], sizeof(vector3<double>));
		of.write((char*)&lines[2], sizeof(vector3<double>));
		of.write((char*)&lines[3], sizeof(vector3<double>));
	}
	else if (ft == 'a'){

	}
	else{

	}
}

bool geo::Rectangle::Collision(ball *ib, unsigned id)
{
	return true;
}

void geo::Plane::Define(vector2<double>& size, vector3<double>& w, vector3<double>& a, vector3<double>& b)
{
	size = size;
	xw = w;
	pa = a - w;
	pb = b - w;
	l1 = pa.length();
	l2 = pb.length();
	u1 = pa / l1;
	u2 = pb / l2;
	uw = u1.cross(u2);
}

bool geo::Plane::Collision(ball* ib, unsigned id)
{
	if (isOnContact){
		vector3<double> dp = ib->Position() - xw;
		vector3<double> wp = vector3<double>(dp.dot(u1), dp.dot(u2), dp.dot(uw));
		if (abs(wp.z) < ib->Radius() && (wp.x > 0 && wp.x < l1) && (wp.y > 0 && wp.y < l2)){
			double dds = -sign(uw.dot(ib->Position() - xw));
			vector3<double> ddd = uw / uw.length();
			vector3<double> unit = dds * ddd;//-sign(uw.dot(ib->Position() - xw)) * (uw / uw.length());
			double cdist = ib->Radius() - abs(unit.dot(ib->Position() - xw));
			if (cdist <= 0){
				std::map<Geometry*, ccontact>::iterator it = ib->ContactWMap().find(this);
				if (it != ib->ContactWMap().end()){
					ib->ContactWMap().erase(this);
				}
				return false;
			}
			std::map<Geometry*, ccontact>::iterator it = ib->ContactWMap().find(this);
			if (it != ib->ContactWMap().end()){
				ccontact* c = &(it->second);
				c->CalculateContactForces(cdist, unit);
			}
			else{
				ccontact c;
				c.SetIBall(ib);
				c.SetJBall(NULL);
				c.SetWall(this);
				c.CalculateContactForces(cdist, unit);
				ib->InsertWContact(this, c);
			}
		}
	}
		/*double t = (ib->Position() - startPoint).dot(ab) / ab.dot();
		if (t < 0.0) t = 0.0;
		if (t > 1.0) t = 1.0;
		vector3<double> d = startPoint + t * ab;
		vector3<double> rp = ib->Position() - d;
		double dist = rp.length();
		double cdist = ib->Radius() - dist;
		if (cdist <= 0){
			std::map<Geometry*, ccontact>::iterator it = ib->ContactWMap().find(this);
			if (it != ib->ContactWMap().end()){
				ib->ContactWMap().erase(this);
			}
			return;
		}
		std::map<Geometry*, ccontact>::iterator it = ib->ContactWMap().find(this);
		if (it != ib->ContactWMap().end()){
			ccontact* c = &(it->second);
			c->CalculateContactForces(cdist, -normal);
		}
		else{
			ccontact c;
			c.SetIBall(ib);
			c.SetJBall(NULL);
			c.SetWall(this);
			c.CalculateContactForces(cdist, -normal);
			ib->InsertWContact(this, c);
		}
	}*/
	return true;
}

void geo::Plane::save2file(std::fstream& pf, char ft)
{

}

void geo::Cube::Define(vector3<double>& sp, vector3<double>& ep)
{
	startPoint = sp;
	endPoint = ep;
	sizex = ep.x - sp.x;
	sizey = ep.y - sp.y;
	sizez = ep.z - sp.z;
	volume = sizex * sizey * sizez;

	Plane* bottom = new Plane(NULL, "bottom", GEO_BOUNDARY);
	bottom->Define(vector2<double>(sizex, sizez), startPoint, startPoint + vector3<double>(0, 0, sizez), startPoint + vector3<double>(sizex, 0, 0));
	planes["bottom"] = bottom;

	Plane* left = new Plane(NULL, "left", GEO_BOUNDARY);
	left->Define(vector2<double>(sizey, sizez), startPoint, startPoint + vector3<double>(0, sizey, 0), startPoint + vector3<double>(0, 0, sizez));
	planes["left"] = left;

	Plane* right = new Plane(NULL, "right", GEO_BOUNDARY);
	right->Define(vector2<double>(sizey, sizez), startPoint + vector3<double>(sizex, 0, 0), startPoint + vector3<double>(sizex, 0, sizez), startPoint + vector3<double>(sizex, sizey, 0));
	planes["right"] = right;

	Plane* back = new Plane(NULL, "back", GEO_BOUNDARY);
	back->Define(vector2<double>(sizex, sizey), startPoint, startPoint + vector3<double>(sizex, 0, 0), startPoint + vector3<double>(0, sizey, 0));
	planes["back"] = back;

	Plane* front = new Plane(NULL, "front", GEO_BOUNDARY);
	front->Define(vector2<double>(sizex, sizey), startPoint + vector3<double>(0, 0, sizez), startPoint + vector3<double>(0, sizey, sizez), startPoint + vector3<double>(sizex, 0, sizez));
	planes["front"] = front;

	Plane* top = new Plane(NULL, "top", GEO_BOUNDARY);
	top->Define(vector2<double>(sizex, sizez), startPoint + vector3<double>(0, sizey, 0), startPoint + vector3<double>(sizex, sizey, 0), startPoint + vector3<double>(0, sizey, sizez));
	planes["top"] = top;
}

void geo::Cube::save2file(std::fstream& of, char ft)
{
	int type = CUBE;
	of.write((char*)&type, sizeof(int));
	int name_size = name.size();
	of.write((char*)&name_size, sizeof(int));
	of.write((char*)name.c_str(), sizeof(char) * name_size);
	vector3<double> vertex[8] = { 0, };
	save_cube_info sci = { startPoint.x, startPoint.y, startPoint.z, sizex, sizey, sizez };
	vertex[0] = startPoint;
	vertex[1] = startPoint + vector3<double>(0, sizey, 0);
	vertex[2] = startPoint + vector3<double>(0, 0, sizez);
	vertex[3] = startPoint + vector3<double>(0, sizey, sizez);
	vertex[4] = startPoint + vector3<double>(sizex, 0, sizez);
	vertex[5] = startPoint + vector3<double>(sizex, sizey, sizez);
	vertex[6] = startPoint + vector3<double>(sizex, 0, 0);
	vertex[7] = startPoint + vector3<double>(sizex, sizey, 0);

	of.write((char*)&sci, sizeof(save_cube_info));
	of.write((char*)vertex, sizeof(double) * 3 * 8);
}

bool geo::Cube::Collision(ball *ib, unsigned id)
{
	if (isOnContact){
		for (std::map<std::string, Plane*>::iterator it = planes.begin(); it != planes.end(); it++){
			it->second->Collision(ib);
		}
	}
	return true;
}

void geo::Cube::SetKnEachPlane()
{
	for (std::map<std::string, Plane*>::iterator it = planes.begin(); it != planes.end(); it++){
		it->second->Kn() = kn;
	}
}

void geo::Shape::Define(std::string fname, vector3<double> pos, material_type mat)
{
	switch (mat){
	case TUNGSTEN_CARBIDE:
		density = TUNGSTEN_CARBIDE_DENSITY;
		youngs = TUNGSTEN_CARBIDE_YOUNGS_MODULUS;
		poisson = TUNGSTEN_CARBIDE_POISSON_RATIO;
		break;
	}
	
	std::cout << "    Define geometry of shape - " << name << " ";
	std::string str;
	std::fstream pf;
	vector3<double> vertex;
	vector3<double> l_vertex;
	vector3<unsigned int> index;
	position = pos;
	pf.open(fname, std::ios::in);

	double max_height = 0;
	while (!pf.eof()){
		pf >> str;
		if (str == "GRID*")
		{
			pf >> str >> vertex.x >> vertex.y >> str >> str >> str >> vertex.z;
			vertex = vertex / 1000.0;
			l_vertex = vertex;
			vertex = l_vertex + pos;
			vertice.push(vertex);
			l_vertice.push(l_vertex);
		}
		if (str == "CTRIA3"){
			pf >> str >> str >> index.x >> index.y >> index.z;
			indice.push(index - vector3<unsigned int>(1, 1, 1));
		}
	}

	vertice.adjustment();
	indice.adjustment();
	std::cout << ".";

	polygon po;
	double t = 0;
	for (vector3<unsigned int>* id = indice.begin(); id != indice.end(); id++)
	{
		po.P = vertice(id->x);
		po.Q = vertice(id->y);
		po.R = vertice(id->z);
		po.V = po.Q - po.P;
		double length = po.V.length();
		po.W = po.R - po.P;
		length = po.W.length();
		po.N = po.V.cross(po.W);
		polygons.push(po);
	}
	polygons.adjustment();
	std::cout << ".";
	id_set = new unsigned[polygons.sizes() * 3];
	unsigned *point_set = new unsigned[polygons.sizes() * 3];
	unsigned int cnt = 0;
	unsigned int id_cnt = 0;
	for (vector3<unsigned int>* id = indice.begin(); id != indice.end(); id++, cnt += 3, id_cnt++){
		id_set[cnt] = id_cnt; point_set[cnt] = id->x;
		id_set[cnt + 1] = id_cnt; point_set[cnt + 1] = id->y;
		id_set[cnt + 2] = id_cnt; point_set[cnt + 2] = id->z;
	}
	std::cout << ".";
	thrust::sort_by_key(point_set, point_set + polygons.sizes() * 3, id_set);
	unsigned int v_size = vertice.sizes();
	poly_start = new unsigned int[v_size];
	poly_end = new unsigned int[v_size];
	cnt = 0;
	for (unsigned int i = 0; i < v_size; i++){
		unsigned int p_id = i;
		poly_start[p_id] = cnt;
		while (point_set[cnt] == p_id)
		{
			cnt++;
		}

		poly_end[p_id] = cnt;
	}
	std::cout << ".";
	delete[] point_set;
	std::cout << " Ok" << std::endl;
}

void geo::Shape::save2file(std::fstream& pf, char ft)
{
	int type = SHAPE;
	pf.write((char*)&type, sizeof(int));
	int name_size = name.size();
	pf.write((char*)&name_size, sizeof(int));;
	pf.write((char*)name.c_str(), sizeof(char) * name_size);
	pf.write((char*)&vertice.sizes(), sizeof(unsigned int));
	pf.write((char*)&indice.sizes(), sizeof(unsigned int));

	pf.write((char*)&position, sizeof(vector3<double>));
	pf.write((char*)l_vertice.get_ptr(), sizeof(vector3<double>) * vertice.sizes());
	pf.write((char*)indice.get_ptr(), sizeof(vector3<unsigned int>) * indice.sizes());
}

bool geo::Shape::Collision(ball *ib, unsigned id)
{
	unsigned int poly_id;
	double dist, cdist;
	polygon poly;
	vector3<double> contact_point, sp, p_unit;
	for (unsigned int i = poly_start[id]; i < poly_end[id]; i++){
		poly_id = id_set[i];
		poly = polygons(poly_id);
		contact_point = ClosestPtPointTriangle(ib->Position(), poly.P, poly.Q, poly.R);
		sp = ib->Position() - contact_point;
		dist = sp.length();
		p_unit = poly.N / poly.N.length();
		cdist = ib->Radius() - dist;

		if (cdist <= 0){
			std::map<Geometry*, ccontact>::iterator it = ib->ContactSMap().find(this);
			if (it != ib->ContactSMap().end()){
				ib->ContactSMap().erase(this);
			}
			return false;
		}
		std::map<Geometry*, ccontact>::iterator it = ib->ContactSMap().find(this);
		if (it != ib->ContactSMap().end()){
			ccontact* c = &(it->second);
			c->CalculateContactForces(cdist, p_unit, kn, ks, fric);
			return true;
		}
		else{
			ccontact c;
		//	std::cout << "particle ID : " << ib->ID() << std::endl;
			c.SetIBall(ib);
			c.SetJBall(NULL);
			c.SetWall(NULL);
			c.SetShape(this);
			c.CalculateContactForces(cdist, p_unit, kn, ks, fric);
			ib->InsertSContact(this, c);
			return true;
		}
	}
	return false;
}

vector3<double> geo::Shape::ClosestPtPointTriangle(vector3<double>& pos, vector3<double>& a, vector3<double>& b, vector3<double>& c)
{
	vector3<double> ab = b - a;
	vector3<double> ac = c - a;
	vector3<double> ap = pos - a;

	double d1 = ab.dot(ap);
	double d2 = ac.dot(ap);
	if (d1 <= 0.0 && d2 <= 0.0){
		//*wc = 0;
		return a;
	}

	vector3<double> bp = pos - b;
	double d3 = ab.dot(bp);
	double d4 = ac.dot(bp);
	if (d3 >= 0.0 && d4 <= d3){
		//*wc = 0;
		return b;
	}
	double vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
		//*wc = 1;
		double v = d1 / (d1 - d3);
		return a + v * ab;
	}

	vector3<double> cp = pos - c;
	double d5 = ab.dot(cp);
	double d6 = ac.dot(cp);
	if (d6 >= 0.0 && d5 <= d6){
		//*wc = 0;
		return c;
	}

	double vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
		//*wc = 1;
		double w = d2 / (d2 - d6);
		return a + w * ac; // barycentric coordinates (1-w, 0, w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	double va = d3 * d6 - d5 * d4;
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
		//*wc = 1;
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric coordinates (0, 1-w, w)
	}
	//*wc = 2;
	// P inside face region. Compute Q through its barycentric coordinates (u, v, w)
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;

	return a + v * ab + w * ac; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

void geo::Shape::Update(double time)
{
	if (!isUpdate)
		return;
	vector3<double> newPosition = func(time);
	//vector3<double> diffCenter = newCenter - position;
	vector3<double> vertex;

	for (unsigned int i = 0; i < vertice.sizes(); i++){
		vertex = newPosition + l_vertice(i);
		vertice(i) = vector3<double>(vertex.x, vertex.y, vertex.z);
	}

	polygon po;
	unsigned int i = 0;

	for (vector3<unsigned int>* id = indice.begin(); id != indice.end(); id++)
	{
		po.P = vertice(id->x);
		po.Q = vertice(id->y);
		po.R = vertice(id->z);
		po.V = po.Q - po.P;
		double length = po.V.length();
		po.W = po.R - po.P;
		length = po.W.length();
		po.N = po.V.cross(po.W);
		polygons(i++) = po;
	}
	isUpdate = true;
	position = newPosition;
}