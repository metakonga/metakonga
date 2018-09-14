#include "vpolygon.h"
#include "numeric_utility.h"
#include "shader.h"
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
	: vobject(_name)
	, vertexList(NULL)
	, indexList(NULL)
	, vertice(NULL)
	, indice(NULL)
	, normals(NULL)
	, colors(NULL)
	, texture(NULL)
	, nvertex(0)
	, ntriangle(0)
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

void vpolygon::_loadSTLASCII(QString f)
{
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
// 	normalList = new double[ntri * 3];
	vertice = new float[ntri * 9];
	normals = new float[ntri * 9];
	double x, y, z;
	float nx, ny, nz;
	qf.reset();
	qts >> ch >> ch >> ch;
	VEC3D p, q, r, c;
	double vol = 0.0;
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
		//vol += numeric::signed_volume_of_triangle(p, q, r);
		c += numeric::utility::calculate_center_of_triangle(p, q, r);
	}
	pos0 = c / ntri;
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
	}
	qf.close();
}

bool vpolygon::define(import_shape_type t, QString file)
{
	switch (t)
	{
	case MILKSHAPE_3D_ASCII: _loadMS3DASCII(file); break;
	case STL_ASCII: _loadSTLASCII(file); break;
	}
	//GLenum check = glewInit();
	origin[0] = 0;// org.x;
	origin[1] = 0; //org.y;
	origin[2] = 0; //org.z;
	ang[0] = 0;
	ang[1] = 0;
	ang[2] = 0;
// 	if (!vertice)
// 		vertice = new VEC3F[nvertex];
// 	if (!colors)
// 		colors = new VEC4F[nvertex];
// 	if (!indice)
// 		indice = new VEC3UI[nindex];
// 	if (!normals)
// 		normals = new VEC3F[nvertex];

// 	for (unsigned int i = 0; i < nvertex; i++){
// 		vertice[i] = VEC3F((float)vset[i].x, (float)vset[i].y, (float(vset[i].z)));
// 		colors[i] = VEC4F(clr.redF(), clr.greenF(), clr.blueF(), 1.0f);
// 		normals[i] = VEC3F((float)nor[i].x, (float)nor[i].y, (float)nor[i].z);
// 	}
// 	for (unsigned int i = 0; i < nindex; i++){
// 		VEC3UI idx = iset[i];
// 		indice[i] = iset[i];
// // 		VEC3F v0 = vertice[idx.y] - vertice[idx.x];
// // 		VEC3F v1 = vertice[idx.z] - vertice[idx.x];
// // 		VEC3F n = v0.cross(v1);
// /*		normals[i] = n / n.length();*/
// 	}

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
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * ntriangle * 9, (float*)vertice, GL_DYNAMIC_DRAW);
	}
	if (!m_normal_vbo)
	{
		glGenBuffers(1, &m_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * ntriangle * 9, (float*)normals, GL_STATIC_DRAW);
		//glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(VEC3F)*nvertex, &(normals[0].x));
		//glBufferSubData(GL_ARRAY_BUFFER, sizeof(VEC3F)*nvt, sizeof(VEC3F) * nid, &(normals[0].x));
	}
 	glBindBuffer(GL_ARRAY_BUFFER, 0);
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	if (!program.Program())
		program.compileProgram(polygonVertexShader, polygonFragmentShader);

	return true;
}

void vpolygon::_drawPolygons()
{
	if (m_vertex_vbo)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vertex_vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*ntriangle * 9, vertice);
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
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*ntriangle * 9, normals);
		}

		//glDrawElements(GL_TRIANGLES, nindex * 3, GL_UNSIGNED_INT, 0);
		glDrawArrays(GL_TRIANGLES, 0, ntriangle * 3);
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
	if (eMode == GL_SELECT)
	{
		glLoadName((GLuint)ID());
	}
	glPushMatrix();
	glTranslated(pos0.x, pos0.y, pos0.z);
	glColor3f(clr.redF(), clr.greenF(), clr.blueF());
	glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
	glUseProgram(program.Program());
	_drawPolygons();
	glUseProgram(0);
	glPopMatrix();
}

void vpolygon::setResultData(unsigned int n)
{

}

void vpolygon::insertResultData(unsigned int i, VEC3D& p, EPD& r)
{

}