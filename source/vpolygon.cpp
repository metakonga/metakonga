#include "vpolygon.h"
#include "shader.h"
#include "vcontroller.h"
#include <QTextStream>

//int vpolygon::pcnt = 1000;

vpolygon::vpolygon()
	//: vglew()
	: vobject()
	, vertice(NULL)
	, indice(NULL)
	, normals(NULL)
	, colors(NULL)
	, texture(NULL)
	, nvertex(0)
	, nindex(0)
	, m_index_vbo(0)
	, m_vertex_vbo(0)
	, m_normal_vbo(0)
	, m_color_vbo(0)
	//, //isSelected(false)
{
	//id = pcnt++;
}

vpolygon::vpolygon(QString& _name)
	//: vglew()
	: vobject(_name)
	, vertice(NULL)
	, indice(NULL)
	, normals(NULL)
	, colors(NULL)
	, texture(NULL)
	, nvertex(0)
	, nindex(0)
	, m_index_vbo(0)
	, m_vertex_vbo(0)
	, m_normal_vbo(0)
	, m_color_vbo(0)
	//, nm(_name)
	//, isSelected(false)
{
	vglew::vglew();
	//id = pcnt++;
}

vpolygon::~vpolygon()
{
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
	float *vertexList = new float[nvert * 3];
	float *textureList = new float[nvert * 2]; 
	int flag, bone;
	float x, y, z, u, v;

	for (unsigned int i = 0; i < nvert; i++)
	{
		qts >> flag >> x >> y >> z >> u >> v >> bone;
		vertexList[i * 3 + 0] = 0.001 * x;
		vertexList[i * 3 + 1] = 0.001 * y;
		vertexList[i * 3 + 2] = 0.001 * z;

		textureList[i * 2 + 0] = u;
		textureList[i * 2 + 1] = v;
	}

	unsigned int nnorm = 0;
	qts >> nnorm;
	float* normalList = new float[nnorm * 3];
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
	texture = new float[nidx * 6];
	unsigned int a, b, c, n1, n2, n3;
	for (unsigned int i = 0; i < nidx; i++)
	{
		qts >> flag >> a >> b >> c >> n1 >> n2 >> n3 >> bone;
		vertice[i * 9 + 0] = vertexList[a * 3 + 0];
		vertice[i * 9 + 1] = vertexList[a * 3 + 1];
		vertice[i * 9 + 2] = vertexList[a * 3 + 2];
		vertice[i * 9 + 3] = vertexList[b * 3 + 0];
		vertice[i * 9 + 4] = vertexList[b * 3 + 1];
		vertice[i * 9 + 5] = vertexList[b * 3 + 2];
		vertice[i * 9 + 6] = vertexList[c * 3 + 0];
		vertice[i * 9 + 7] = vertexList[c * 3 + 1];
		vertice[i * 9 + 8] = vertexList[c * 3 + 2];

		normals[i * 9 + 0] = normalList[n1 * 3 + 0];
		normals[i * 9 + 1] = normalList[n1 * 3 + 1];
		normals[i * 9 + 2] = normalList[n1 * 3 + 2];
		normals[i * 9 + 3] = normalList[n2 * 3 + 0];
		normals[i * 9 + 4] = normalList[n2 * 3 + 1];
		normals[i * 9 + 5] = normalList[n2 * 3 + 2];
		normals[i * 9 + 6] = normalList[n3 * 3 + 0];
		normals[i * 9 + 7] = normalList[n3 * 3 + 1];
		normals[i * 9 + 8] = normalList[n3 * 3 + 2];

		texture[i * 6 + 0] = textureList[a * 2 + 0];
		texture[i * 6 + 1] = textureList[a * 2 + 1];
		texture[i * 6 + 2] = textureList[b * 2 + 0];
		texture[i * 6 + 3] = textureList[b * 2 + 1];
		texture[i * 6 + 4] = textureList[c * 2 + 0];
		texture[i * 6 + 5] = textureList[c * 2 + 1];
	}
	nvertex = nvert;
	nindex = nidx;
	delete[] vertexList;
	delete[] normalList;
	delete[] textureList;
	qf.close();
}

bool vpolygon::define(import_shape_type t, QString file)
{
	switch (t)
	{
	case MILKSHAPE_3D_ASCII: _loadMS3DASCII(file); break;
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
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nindex * 9, (float*)vertice, GL_DYNAMIC_DRAW);
	}
// 	if (!m_index_vbo){
// 		glGenBuffers(1, &m_index_vbo);
// 		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_vbo);
// 		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * nindex * 3, (unsigned int*)indice, GL_STATIC_DRAW);
// 	}
// 	if (!m_color_vbo)
// 	{
// 		glGenBuffers(1, &m_color_vbo);
// 		glBindBuffer(GL_ARRAY_BUFFER, m_color_vbo);
// 		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nvertex * 4, (float*)colors, GL_STATIC_DRAW);
// 	}
	if (!m_normal_vbo)
	{
		glGenBuffers(1, &m_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nindex * 9, (float*)normals, GL_STATIC_DRAW);
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
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*nindex * 9, vertice);
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
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*nindex * 9, normals);
		}

		//glDrawElements(GL_TRIANGLES, nindex * 3, GL_UNSIGNED_INT, 0);
		glDrawArrays(GL_TRIANGLES, 0, nindex * 3);
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

	glColor3f(clr.redF(), clr.greenF(), clr.blueF());
//	glColor3f(0.0f, 0.0f, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
	glUseProgram(program.Program());
	//glUniform1f(glGetUniformLocation(program.Program(), "lightDir"), )
	_drawPolygons();
	glUseProgram(0);

// 	if (eMode == GL_SELECT) 
// 		glLoadName((GLuint)ID());
// 
// 	glPushMatrix();
// 	if (vcontroller::getFrame() && outPos && outRot)
// 	{
// 		//glPushMatrix();
// 		unsigned int f = vcontroller::getFrame();
// 		glTranslated(outPos[f].x, outPos[f].y, outPos[f].z);
// 		VEC3D e;// = ep2e(outRot[f]);
// 		double xi = (e.x * 180) / M_PI;// +ang[0];
// 		double th = (e.y * 180) / M_PI;//; +ang[1];
// 		double ap = (e.z * 180) / M_PI;// +ang[2];
// 		glRotated(xi/* - ang[0]*/, 0, 0, 1);
// 		glRotated(th/* - ang[1]*/, 1, 0, 0);
// 		glRotated(ap/* - ang[2]*/, 0, 0, 1);
// 		/*glPopMatrix();*/
// 		//glCallList(coord);
// 	}
// 	else{
// 		glTranslated(origin[0], origin[1], origin[2]);
// 		//glPushMatrix();
// 		glRotated(ang[0], 0, 0, 1);
// 		glRotated(ang[1], 1, 0, 0);
// 		glRotated(ang[2], 0, 0, 1);
// 	}
// 	
// 	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
// 	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
// 	//glBindBuffer(GL_ARRAY_BUFFER, m_color_vbo);
// 	//glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_vbo);
// 
// 	
// 	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_vbo);
// 	//glIndexPointer(GL_UNSIGNED_INT, 0, 0);
// 
// 	glEnableClientState(GL_VERTEX_ARRAY);
// 	//glEnableClientState(GL_NORMAL_ARRAY);
// //	glEnableClientState(GL_COLOR_ARRAY);
// 	//glEnableClientState(GL_NORMAL_ARRAY);
// 
// 	//glNormalPointer(GL_FLOAT, 0, 0);
// 	glVertexPointer(3, GL_FLOAT, 0, 0);
// 	//glColorPointer(3, GL_FLOAT, 0, 0);
// 	glDrawElements(GL_TRIANGLES, nindex*3, GL_UNSIGNED_INT, 0);
// 
// 	//glDrawArrays(GL_TRIANGLES, 0, 5);
// 	glDisableClientState(GL_VERTEX_ARRAY);
// 	//glDisableClientState(GL_NORMAL_ARRAY);
// 	//glDisableClientState(GL_COLOR_ARRAY);
// 	//glDisableClientState(GL_NORMAL_ARRAY);
// 
// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
// 	//glBindBuffer(GL_ARRAY_BUFFER, 0);
// 	//glBindBuffer(GL_ARRAY_BUFFER, 0);
// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
// 
// 	if (spheres)
// 	{
// // 		glColor3f(1.f, 1.f, 1.f);
// // 		for (unsigned int i = 0; i < nindex; i++){
// // 			glPushMatrix();
// // 			glTranslatef(spheres[i].x, spheres[i].y, spheres[i].z);
// // 			glutSolidSphere(spheres[i].w, 16, 16);
// // 			glPopMatrix();
// // 		}
// 	}
// 	glPopMatrix();
// 	//mmglPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void vpolygon::setResultData(unsigned int n)
{

}

void vpolygon::insertResultData(unsigned int i, VEC3D& p, EPD& r)
{

}