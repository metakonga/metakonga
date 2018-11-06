#include "vpolygon.h"
#include "numeric_utility.h"
#include "shader.h"
#include "model.h"
#include "vcube.h"
#include "vcontroller.h"
#include <QTextStream>


//int vpolygon::pcnt = 1000;

vpolygon::vpolygon()
	//: vglew()
	: vobject()
	, vertexList(NULL)
	, indexList(NULL)
	, vertice(NULL)
	, indice(NULL)
	, normals(NULL)
	, colors(NULL)
	, texture(NULL)
	, nvertex(0)
	, ntriangle(0)
	, nvtriangle(0)
	, m_index_vbo(0)
	, m_vertex_vbo(0)
	, m_normal_vbo(0)
	, m_color_vbo(0)
	//, //isSelected(false)
{
	vobject::vot = GEOMETRY_OBJECT;
}

vpolygon::vpolygon(QString& _name)
	//: vglew()
	: vobject(V_POLYGON, _name)
	, vertexList(NULL)
	, indexList(NULL)
	, vertice(NULL)
	, indice(NULL)
	, normals(NULL)
	, colors(NULL)
	, texture(NULL)
	, nvertex(0)
	, ntriangle(0)
	, nvtriangle(0)
	, m_index_vbo(0)
	, m_vertex_vbo(0)
	, m_normal_vbo(0)
	, m_color_vbo(0)
	//, nm(_name)
	//, isSelected(false)
{
	vglew::vglew();
	vobject::vot = GEOMETRY_OBJECT;
	//id = pcnt++;
}

vpolygon::~vpolygon()
{
	if (vertexList) delete[] vertexList; vertexList = NULL;
	if (indexList) delete[] indexList; indexList = NULL;
	if (vertice) delete[] vertice; vertice = NULL;
	if (indice) delete[] indice; indice = NULL;
	if (normals) delete[] normals; normals = NULL;
	if (colors) delete[] colors; colors = NULL;
	if (texture) delete[] texture; texture = NULL;
	if (select_cube) delete[] select_cube; select_cube = NULL;
	if (m_vertex_vbo){
		glDeleteBuffers(1, &m_vertex_vbo);
		m_vertex_vbo = 0;
	}
	if (m_index_vbo){
		glDeleteBuffers(1, &m_index_vbo);
		m_index_vbo = 0;
	}
	if (m_color_vbo){
		glDeleteBuffers(1, &m_color_vbo);
		m_color_vbo = 0;
	}
	if (m_normal_vbo){
		glDeleteBuffers(1, &m_normal_vbo);
		m_normal_vbo = 0;
	}
}

bool vpolygon::makePolygonGeometry(VEC3F& P, VEC3F& Q, VEC3F& R)
{
// 	p[0] = P.x; p[1] = P.y; p[2] = P.z;
// 	q[0] = Q.x; q[1] = Q.y; q[2] = Q.z;
// 	r[0] = R.x; r[1] = R.y; r[2] = R.z;

//	this->define();
	return true;
}

void vpolygon::_loadMS3DASCII(QString f)
{
	QFile qf(f);
	qf.open(QIODevice::ReadOnly);
	QTextStream qts(&qf);
	QString ch;
	//unsigned int nvertex = 0;
	//unsigned int npoly = 0;
	qts >> ch >> ch >> ch >> ch >> ch >> ch >> ch >> ch >> ch >> ch;
	unsigned int ui;
	qts >> ch;
	int begin = ch.indexOf("\"");
	int end = ch.lastIndexOf("\"");
	QString _name = ch.mid(begin + 1, end - 1);
	unsigned int nvert = 0;
	qts >> ch >> ch >> nvert;
	vertexList = new double[nvert * 3];
	double* textureList = new double[nvert * 2];
	int flag, bone;
	double x, y, z, u, v;
	//float cx = 0, cy = 0, cz = 0;
	for (unsigned int i = 0; i < nvert; i++)
	{
		qts >> flag >> x >> y >> z >> u >> v >> bone;
		vertexList[i * 3 + 0] = 0.001 * x;
		vertexList[i * 3 + 1] = 0.001 * y;
		vertexList[i * 3 + 2] = 0.001 * z;
// 		cx += x;
// 		cy += y;
// 		cz += z;
		textureList[i * 2 + 0] = u;
		textureList[i * 2 + 1] = v;
	}
// 	origin[0] = 0.001 * cx / nvert;
// 	origin[1] = 0.001 * cy / nvert;
// 	origin[2] = 0.001 * cz / nvert;
	unsigned int nnorm = 0;
	qts >> nnorm;
	double* normalList = new double[nnorm * 3];
	for (unsigned int i = 0; i < nnorm; i++)
	{
		qts >> x >> y >> z;
		normalList[i * 3 + 0] = x;
		normalList[i * 3 + 1] = y;
		normalList[i * 3 + 2] = z;
	}
	unsigned int nidx = 0;
	qts >> nidx;
	vertice = new float[nidx * 9];
	normals = new float[nidx * 9];
//	texture = new float[nidx * 6];
	indexList = new unsigned int[nidx * 3];
	unsigned int a, b, c, n1, n2, n3;
	for (unsigned int i = 0; i < nidx; i++)
	{
		qts >> flag >> a >> b >> c >> n1 >> n2 >> n3 >> bone;
		indexList[i * 3 + 0] = a;
		indexList[i * 3 + 1] = b;
		indexList[i * 3 + 2] = c;
		vertice[i * 9 + 0] = (float)vertexList[a * 3 + 0];
		vertice[i * 9 + 1] = (float)vertexList[a * 3 + 1];
		vertice[i * 9 + 2] = (float)vertexList[a * 3 + 2];
		vertice[i * 9 + 3] = (float)vertexList[b * 3 + 0];
		vertice[i * 9 + 4] = (float)vertexList[b * 3 + 1];
		vertice[i * 9 + 5] = (float)vertexList[b * 3 + 2];
		vertice[i * 9 + 6] = (float)vertexList[c * 3 + 0];
		vertice[i * 9 + 7] = (float)vertexList[c * 3 + 1];
		vertice[i * 9 + 8] = (float)vertexList[c * 3 + 2];

		normals[i * 9 + 0] = (float)normalList[n1 * 3 + 0];
		normals[i * 9 + 1] = (float)normalList[n1 * 3 + 1];
		normals[i * 9 + 2] = (float)normalList[n1 * 3 + 2];
		normals[i * 9 + 3] = (float)normalList[n2 * 3 + 0];
		normals[i * 9 + 4] = (float)normalList[n2 * 3 + 1];
		normals[i * 9 + 5] = (float)normalList[n2 * 3 + 2];
		normals[i * 9 + 6] = (float)normalList[n3 * 3 + 0];
		normals[i * 9 + 7] = (float)normalList[n3 * 3 + 1];
		normals[i * 9 + 8] = (float)normalList[n3 * 3 + 2];

// 		texture[i * 6 + 0] = textureList[a * 2 + 0];
// 		texture[i * 6 + 1] = textureList[a * 2 + 1];
// 		texture[i * 6 + 2] = textureList[b * 2 + 0];
// 		texture[i * 6 + 3] = textureList[b * 2 + 1];
// 		texture[i * 6 + 4] = textureList[c * 2 + 0];
// 		texture[i * 6 + 5] = textureList[c * 2 + 1];
	}
	nvertex = nvert;
	ntriangle = nidx;
	delete[] normalList;
	delete[] textureList;
	qf.close();
}

void vpolygon::_loadSTLASCII(QString f, double lx, double ly, double lz)
{
	file_path = f;
	QFile qf(f);
	qf.open(QIODevice::ReadOnly);
	QTextStream qts(&qf);
	QString ch;
	//unsigned int nvertex = 0;
	//unsigned int npoly = 0;
	qts >> ch >> ch >> ch;// >> ch >> ch >> ch >> ch >> ch >> ch >> ch;
	unsigned int ntri = 0;
	while (!qts.atEnd())
	{
		qts >> ch;
		if (ch == "facet")
			ntri++;
	}
 	vertexList = new double[ntri * 9];
//	vertexList = new float[ntri * 9];
// 	normalList = new double[ntri * 3];
	vertice = new float[ntri * 9];
	normals = new float[ntri * 9];
	pos0 = VEC3D(lx, ly, lz);
	double x, y, z;
	float nx, ny, nz;
	qf.reset();
	qts >> ch >> ch >> ch;
	VEC3D p, q, r, c;
	double _vol = 0.0;
	double min_radius = 10000.0;
	double max_radius = 0.0;
	min_point = FLT_MAX;
	max_point = FLT_MIN;
	VEC3D *spos = new VEC3D[ntri];
	ixx = iyy = izz = ixy = ixz = iyz = 0.0;
	//unsigned int nc = 0;
	for (unsigned int i = 0; i < ntri; i++)
	{
		qts >> ch >> ch >> nx >> ny >> nz;
		normals[i * 9 + 0] = nx;
		normals[i * 9 + 1] = ny;
		normals[i * 9 + 2] = nz;
		normals[i * 9 + 3] = nx;
		normals[i * 9 + 4] = ny;
		normals[i * 9 + 5] = nz;
		normals[i * 9 + 6] = nx;
		normals[i * 9 + 7] = ny;
		normals[i * 9 + 8] = nz;
		qts >> ch >> ch;
		qts >> ch >> x >> y >> z;
		p.x = vertexList[i * 9 + 0] = 0.001 * x;
		p.y = vertexList[i * 9 + 1] = 0.001 * y;
		p.z = vertexList[i * 9 + 2] = 0.001 * z;
		vertice[i * 9 + 0] = (float)vertexList[i * 9 + 0];
		vertice[i * 9 + 1] = (float)vertexList[i * 9 + 1];
		vertice[i * 9 + 2] = (float)vertexList[i * 9 + 2];

		qts >> ch >> x >> y >> z;
		q.x = vertexList[i * 9 + 3] = 0.001 * x;
		q.y = vertexList[i * 9 + 4] = 0.001 * y;
		q.z = vertexList[i * 9 + 5] = 0.001 * z;
		vertice[i * 9 + 3] = (float)vertexList[i * 9 + 3];
		vertice[i * 9 + 4] = (float)vertexList[i * 9 + 4];
		vertice[i * 9 + 5] = (float)vertexList[i * 9 + 5];

		qts >> ch >> x >> y >> z;
		r.x = vertexList[i * 9 + 6] = 0.001 * x;
		r.y = vertexList[i * 9 + 7] = 0.001 * y;
		r.z = vertexList[i * 9 + 8] = 0.001 * z;
		vertice[i * 9 + 6] = (float)vertexList[i * 9 + 6];
		vertice[i * 9 + 7] = (float)vertexList[i * 9 + 7];
		vertice[i * 9 + 8] = (float)vertexList[i * 9 + 8];
		qts >> ch >> ch;
		_vol += numeric::utility::signed_volume_of_triangle(p, q, r);
		spos[i] = numeric::utility::calculate_center_of_triangle(p, q, r);
		//c += p + q + r;// spos[i];
		//nc += 3;
		double _r = (spos[i] - p).length();
		if (max_radius < _r)
			max_radius = _r;
		if (min_radius > _r)
			min_radius = _r;
		min_point.x = min(numeric::utility::getMinValue(p.x, q.x, r.x), min_point.x);
		min_point.y = min(numeric::utility::getMinValue(p.y, q.y, r.y), min_point.y);
		min_point.z = min(numeric::utility::getMinValue(p.z, q.z, r.z), min_point.z);
		max_point.x = max(numeric::utility::getMaxValue(p.x, q.x, r.x), max_point.x);
		max_point.y = max(numeric::utility::getMaxValue(p.y, q.y, r.y), max_point.y);
		max_point.z = max(numeric::utility::getMaxValue(p.z, q.z, r.z), max_point.z);
		
	}
#ifdef _DEBUG
	qDebug() << "Maximum radius of " << name() << " is " << max_radius;
	qDebug() << "Minimum radius of " << name() << " is " << min_radius;
#endif
	//pos0 = c / nc;
	ntriangle = ntri;
	for (unsigned int i = 0; i < ntri; i++)
	{
		int s = i * 9;
		vertice[s + 0] -= pos0.x;
		vertice[s + 1] -= pos0.y;
		vertice[s + 2] -= pos0.z;
		vertice[s + 3] -= pos0.x;
		vertice[s + 4] -= pos0.y;
		vertice[s + 5] -= pos0.z;
		vertice[s + 6] -= pos0.x;
		vertice[s + 7] -= pos0.y;
		vertice[s + 8] -= pos0.z;
		VEC3D cm = spos[i] - pos0;
		ixx += cm.y * cm.y + cm.z * cm.z;
		iyy += cm.x * cm.x + cm.z * cm.z;
		izz += cm.x * cm.x + cm.y * cm.y;
		ixy -= cm.x * cm.y;
		ixz -= cm.x * cm.z;
		iyz -= cm.y * cm.z;
	}
	qf.close();
	vol = _vol;
	delete[] spos;
	nvtriangle = ntriangle;
	//splitTriangle(0.001);
}

QList<triangle_info> vpolygon::_splitTriangle(triangle_info& ti, double to)
{
	QList<triangle_info> added_tri;
	QList<triangle_info> temp_tri;
	//ati.push_back(ti);
	bool isAllDone = false;
	while (!isAllDone)
	{
		isAllDone = true;
		QList<triangle_info> ati;
		if (temp_tri.size())
		{
			ati = temp_tri;
			temp_tri.clear();
		}
		else
			ati.push_back(ti);
		foreach(triangle_info t, ati)
		{
			if (t.rad > to)
			{
				isAllDone = false;
				int tid = 0;
				VEC3D midp;
				double s_pq = (t.q - t.p).length();
				double s_qr = (t.r - t.q).length();
				double s_pr = (t.r - t.p).length();
				if (s_pq > s_qr)
				{
					if (s_pq > s_pr)
					{
						midp = 0.5 * (t.q + t.p);
						tid = 3;
					}
					else
					{
						midp = 0.5 * (t.r + t.p);
						tid = 2;
					}
				}
				else
				{
					if (s_qr > s_pr)
					{
						midp = 0.5 * (t.r + t.q);
						tid = 1;
					}
					else
					{
						midp = 0.5 * (t.r + t.p);
						tid = 2;
					}
				}
				VEC3D aspos;//double aspos = 0.0;
				double arad = 0.0;
				VEC3D an;
				VEC3D p = t.p;
				VEC3D q = t.q;
				VEC3D r = t.r;
				if (tid == 1)
				{
					aspos = numeric::utility::calculate_center_of_triangle(p, q, midp);
					an = (q - p).cross(midp - p);
					an = an / an.length();
					arad = (p - aspos).length();
					triangle_info ati0 = { arad, p, q, midp, an };
					aspos = numeric::utility::calculate_center_of_triangle(p, midp, r);
					an = (midp - p).cross(r - p);
					an = an / an.length();
					arad = (p - aspos).length();
					triangle_info ati1 = { arad, p, midp, r, an };
					temp_tri.push_back(ati0);
					temp_tri.push_back(ati1);
				}
				else if (tid == 2)
				{
					aspos = numeric::utility::calculate_center_of_triangle(q, r, midp);
					an = (r - q).cross(midp - q);
					an = an / an.length();
					arad = (q - aspos).length();
					triangle_info ati0 = { arad, q, r, midp, an };
					aspos = numeric::utility::calculate_center_of_triangle(q, midp, p);
					an = (midp - q).cross(p - q);
					an = an / an.length();
					arad = (q - aspos).length();
					triangle_info ati1 = { arad, q, midp, p, an };
					temp_tri.push_back(ati0);
					temp_tri.push_back(ati1);
				}
				else if (tid == 3)
				{
					aspos = numeric::utility::calculate_center_of_triangle(r, p, midp);
					an = (p - r).cross(midp - r);
					an = an / an.length();
					arad = (r - aspos).length();
					triangle_info ati0 = { arad, r, p, midp, an };
					aspos = numeric::utility::calculate_center_of_triangle(r, midp, q);
					an = (midp - r).cross(q - r);
					an = an / an.length();
					arad = (r - aspos).length();
					triangle_info ati1 = { arad, r, midp, q, an };
					temp_tri.push_back(ati0);
					temp_tri.push_back(ati1);
				}
			}
			else
			{
				added_tri.push_back(t);
			}
		}
	}	
	return added_tri;
}

void vpolygon::splitTriangle(double to)
{
	VEC3D p, q, r, n;
	
	QList<triangle_info> temp_tri;
	for (unsigned int i = 0; i < ntriangle; i++)
	{
		int  s = i * 9;
		p = VEC3D(vertexList[s + 0], vertexList[s + 1], vertexList[s + 2]);
		q = VEC3D(vertexList[s + 3], vertexList[s + 4], vertexList[s + 5]);
		r = VEC3D(vertexList[s + 6], vertexList[s + 7], vertexList[s + 8]);
		VEC3D spos = numeric::utility::calculate_center_of_triangle(p, q, r);
		double rad = (spos - p).length();
		triangle_info tinfo = { rad, p, q, r, VEC3D(normals[s + 0], normals[s + 1], normals[s + 2]) };	
		if (rad > to)
		{
			QList<triangle_info> added_tri = _splitTriangle(tinfo, to);
			foreach(triangle_info t, added_tri)
			{
				temp_tri.push_back(t);
			}
		}
		else
		{
			temp_tri.push_back(tinfo);
		}
	}
	//delete[] vertice;
	//delete[] normals;
	delete[] vertexList;
	ntriangle = temp_tri.size();
	//vertice = new float[ntriangle * 9];
	//normals = new float[ntriangle * 9];
	vertexList = new double[ntriangle * 9];
	int cnt = 0;
	foreach(triangle_info t, temp_tri)
	{
		int s = cnt * 9;
		vertexList[s + 0] = t.p.x;
		vertexList[s + 1] = t.p.y;
		vertexList[s + 2] = t.p.z;
		vertexList[s + 3] = t.q.x;
		vertexList[s + 4] = t.q.y;
		vertexList[s + 5] = t.q.z;
		vertexList[s + 6] = t.r.x;
		vertexList[s + 7] = t.r.y;
		vertexList[s + 8] = t.r.z;

// 		vertice[s + 0] = (float)t.p.x - pos0.x;
// 		vertice[s + 1] = (float)t.p.y - pos0.y;
// 		vertice[s + 2] = (float)t.p.z - pos0.z;
// 		vertice[s + 3] = (float)t.q.x - pos0.x;
// 		vertice[s + 4] = (float)t.q.y - pos0.y;
// 		vertice[s + 5] = (float)t.q.z - pos0.z;
// 		vertice[s + 6] = (float)t.r.x - pos0.x;
// 		vertice[s + 7] = (float)t.r.y - pos0.y;
// 		vertice[s + 8] = (float)t.r.z - pos0.z;

// 		normals[s + 0] = t.n.x;
// 		normals[s + 1] = t.n.y;
// 		normals[s + 2] = t.n.z;
// 		normals[s + 3] = t.n.x;
// 		normals[s + 4] = t.n.y;
// 		normals[s + 5] = t.n.z;
// 		normals[s + 6] = t.n.x;
// 		normals[s + 7] = t.n.y;
// 		normals[s + 8] = t.n.z;
		cnt++;
	}
}

bool vpolygon::define(import_shape_type t, QString file, double x, double y, double z)
{
	ist = t;
	switch (t)
	{
	case MILKSHAPE_3D_ASCII: _loadMS3DASCII(file); break;
	case STL_ASCII: _loadSTLASCII(file, x, y, z); break;
	}
	//GLenum check = glewInit();
	origin[0] = 0;// org.x;
	origin[1] = 0; //org.y;
	origin[2] = 0; //org.z;
	ang[0] = 0;
	ang[1] = 0;
	ang[2] = 0;

	vcube *vc = new vcube(QString(""));
	VEC3D sz = max_point - min_point;
	vc->makeCubeGeometry(QString(""), NO_USE_TYPE, NO_MATERIAL, min_point.To<float>(), sz.To<float>());
	vc->setColor(RED);
	select_cube = vc;

	if (m_vertex_vbo){
		glDeleteBuffers(1, &m_vertex_vbo);
		m_vertex_vbo = 0;
	}
	if (m_index_vbo){
		glDeleteBuffers(1, &m_index_vbo);
		m_index_vbo = 0;
	}
	if (m_color_vbo){
		glDeleteBuffers(1, &m_color_vbo);
		m_color_vbo = 0;
	}
	if (m_normal_vbo){
		glDeleteBuffers(1, &m_normal_vbo);
		m_normal_vbo = 0;
	}

	if (!m_vertex_vbo)
	{
		glGenBuffers(1, &m_vertex_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nvtriangle * 9, (float*)vertice, GL_DYNAMIC_DRAW);
	}
	if (!m_normal_vbo)
	{
		glGenBuffers(1, &m_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nvtriangle * 9, (float*)normals, GL_STATIC_DRAW);
		//glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(VEC3F)*nvertex, &(normals[0].x));
		//glBufferSubData(GL_ARRAY_BUFFER, sizeof(VEC3F)*nvt, sizeof(VEC3F) * nid, &(normals[0].x));
	}
// 	if (!m_color_vbo)
// 	{
// 		glGenBuffers(1, &m_color_vbo);
// 		glBindBuffer(GL_ARRAY_BUFFER, m_color_vbo);
// 		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nvertex * 4, (float*)colors, GL_STATIC_DRAW);
// 		//glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(VEC3F)*nvertex, &(normals[0].x));
// 		//glBufferSubData(GL_ARRAY_BUFFER, sizeof(VEC3F)*nvt, sizeof(VEC3F) * nid, &(normals[0].x));
// 	}
 	glBindBuffer(GL_ARRAY_BUFFER, 0);
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (!program.Program())
		program.compileProgram(polygonVertexShader, polygonFragmentShader);
	display = true;
	return true;
}

void vpolygon::_drawPolygons()
{
	GLfloat ucolor[4] = { clr.redF(), clr.greenF(), clr.blueF(), clr.alphaF() };
	int loc_color = glGetUniformLocation(program.Program(), "ucolor");
	glUniform4fv(loc_color, 1, ucolor);
	if (m_vertex_vbo)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vertex_vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*nvtriangle * 9, vertice);
// 		if (m_index_vbo)
// 		{
// 			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_vbo);
// 		}
// 		if (m_color_vbo)
// 		{
// 			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_color_vbo);
// 			glColorPointer(4, GL_FLOAT, 0, 0);
// 			glEnableClientState(GL_COLOR_ARRAY);
// 			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*nvertex * 4, colors);
// 		}
		if (m_normal_vbo)
		{
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_normal_vbo);
			glNormalPointer(GL_FLOAT, 0, 0);
			glEnableClientState(GL_NORMAL_ARRAY);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*nvtriangle * 9, normals);
		}

		//glDrawElements(GL_TRIANGLES, nindex * 3, GL_UNSIGNED_INT, 0);
		glDrawArrays(GL_TRIANGLES, 0, nvtriangle * 3);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		//glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);
		if(m_vertex_vbo)
			glDisableClientState(GL_VERTEX_ARRAY);
		if(m_color_vbo)
			glDisableClientState(GL_COLOR_ARRAY);
		if (m_normal_vbo)
			glDisableClientState(GL_NORMAL_ARRAY);
	}
}

void vpolygon::draw(GLenum eMode)
{
	if (display)
	{
		if (isSelected)
		{
			select_cube->draw(GL_RENDER);
		}
		if (eMode == GL_SELECT)
		{
			glLoadName((GLuint)ID());
		}
		glPushMatrix();
		unsigned int idx = vcontroller::getFrame();
		if (idx != 0)
		{
			if (model::rs->pointMassResults().find(nm) != model::rs->pointMassResults().end())
			{
				VEC3D p = model::rs->pointMassResults()[nm].at(idx).pos;
			//	qDebug() << p.x << " " << p.y << " " << p.z;
				EPD ep = model::rs->pointMassResults()[nm].at(idx).ep;
				animationFrame(p, ep);// p.x, p.y, p.z);
			}
			else
			{
				glTranslated(pos0.x, pos0.y, pos0.z);
				glRotated(ang0.x, 0, 0, 1);
				glRotated(ang0.y, 1, 0, 0);
				glRotated(ang0.z, 0, 0, 1);
			}				
		}
		else
		{
			glTranslated(pos0.x, pos0.y, pos0.z);
			glRotated(ang0.x, 0, 0, 1);
			glRotated(ang0.y, 1, 0, 0);
			glRotated(ang0.z, 0, 0, 1);
		}			
	

		glColor3f(1.0f, 0.0f, 0.0f);
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glUseProgram(program.Program());
		_drawPolygons();
		glUseProgram(0);
		glPopMatrix();
	}
	
}

void vpolygon::setResultData(unsigned int n)
{

}

void vpolygon::insertResultData(unsigned int i, VEC3D& p, EPD& r)
{

}