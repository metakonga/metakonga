#include "vpolygon.h"
#include "vcontroller.h"

int vpolygon::pcnt = 1000;

vpolygon::vpolygon()
	//: vglew()
	: vertice(NULL)
	, indice(NULL)
	, normals(NULL)
	, spheres(NULL)
	, outPos(NULL)
	, outRot(NULL)
	, nvertex(0)
	, nindex(0)
	, m_index_vbo(0)
	, m_vertex_vbo(0)
{
	id = pcnt++;
}

vpolygon::vpolygon(QString& _name)
	//: vglew()
	: vertice(NULL)
	, indice(NULL)
	, normals(NULL)
	, spheres(NULL)
	, outPos(NULL)
	, outRot(NULL)
	, nvertex(0)
	, nindex(0)
	, m_index_vbo(0)
	, m_vertex_vbo(0)
	, nm(_name)
{
	id = pcnt++;
}

vpolygon::~vpolygon()
{
	if (vertice) delete[] vertice; vertice = NULL;
	if (indice) delete[] indice; indice = NULL;
	if (normals) delete[] normals; normals = NULL;
	if (spheres) delete[] spheres; spheres = NULL;
	if (outPos) delete[] outPos; outPos = NULL;
	if (outRot) delete[] outRot; outRot = NULL;

	if (m_vertex_vbo){
		glDeleteBuffers(1, &m_vertex_vbo);
		m_vertex_vbo = 0;
	}
	if (m_index_vbo){
		glDeleteBuffers(1, &m_index_vbo);
		m_index_vbo = 0;
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

bool vpolygon::define(VEC3D org, host_polygon_info* hpi, VEC4D* _sphere, VEC3D* vset, VEC3UI* iset, unsigned int nid, unsigned int nvt)
{
	//GLenum check = glewInit();
	origin[0] = org.x;
	origin[1] = org.y;
	origin[2] = org.z;
	ang[0] = 0;
	ang[1] = 0;
	ang[2] = 0;
	nindex = nid;
	nvertex = nvt;
	if (!vertice)
		vertice = new VEC3F[nvertex];
	if (!indice)
		indice = new VEC3UI[nindex];
	if (!normals)
		normals = new VEC3F[nindex];
	if (!spheres)
		spheres = new VEC4F[nindex];

	for (unsigned int i = 0; i < nvertex; i++){
		vertice[i] = vset[i].To<float>();
	}
	for (unsigned int i = 0; i < nindex; i++){
		indice[i] = iset[i];
		normals[i] = hpi[i].N.To<float>();
		spheres[i] = _sphere[i].To<float>();
	}

	if (m_vertex_vbo){
		glDeleteBuffers(1, &m_vertex_vbo);
		m_vertex_vbo = 0;
	}
	if (m_index_vbo){
		glDeleteBuffers(1, &m_index_vbo);
		m_index_vbo = 0;
	}

	if (!m_vertex_vbo)
	{
		glGenBuffers(1, &m_vertex_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(VEC3F)*nvertex/* + sizeof(VEC3F)*nid*/, 0, GL_DYNAMIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(VEC3F)*nvertex, &(vertice[0].x));
		//glBufferSubData(GL_ARRAY_BUFFER, sizeof(VEC3F)*nvt, sizeof(VEC3F) * nid, &(normals[0].x));
	}
	//m_vertex_vbo = vglew::createVBO<float>(sizeof(VEC3D) * nvt, &(vertice[0].x));
	if (!m_index_vbo){
		glGenBuffers(1, &m_index_vbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_vbo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(VEC3UI) * nindex, &(indice[0].x), GL_STATIC_DRAW);
	}
		//m_index_vbo = vglew::createVBO<unsigned int>(sizeof(VEC3UI) * nid, &(indice[0].x));

	return true;
}

void vpolygon::draw(GLenum eMode)
{
	if (eMode == GL_SELECT) glLoadName((GLuint)ID());
	glPushMatrix();
	if (vcontroller::getFrame() && outPos && outRot)
	{
		//glPushMatrix();
		unsigned int f = vcontroller::getFrame();
		glTranslated(outPos[f].x, outPos[f].y, outPos[f].z);
		VEC3D e = ep2e(outRot[f]);
		double xi = (e.x * 180) / M_PI;// +ang[0];
		double th = (e.y * 180) / M_PI;//; +ang[1];
		double ap = (e.z * 180) / M_PI;// +ang[2];
		glRotated(xi/* - ang[0]*/, 0, 0, 1);
		glRotated(th/* - ang[1]*/, 1, 0, 0);
		glRotated(ap/* - ang[2]*/, 0, 0, 1);
		/*glPopMatrix();*/
		//glCallList(coord);
	}
	else{
		glTranslated(origin[0], origin[1], origin[2]);
		//glPushMatrix();
		glRotated(ang[0], 0, 0, 1);
		glRotated(ang[1], 1, 0, 0);
		glRotated(ang[2], 0, 0, 1);
	}
	
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_vbo);

	//glNormalPointer(GL_FLOAT, 0, 0);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_vbo);
	//glIndexPointer(GL_UNSIGNED_INT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_NORMAL_ARRAY);
	glDrawElements(GL_TRIANGLES, nindex*3, GL_UNSIGNED_INT, 0);
	//glDrawArrays(GL_TRIANGLES, 0, 5);
	glDisableClientState(GL_VERTEX_ARRAY);
	//glDisableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	
	if (spheres)
	{
// 		glColor3f(1.f, 1.f, 1.f);
// 		for (unsigned int i = 0; i < nindex; i++){
// 			glPushMatrix();
// 			glTranslatef(spheres[i].x, spheres[i].y, spheres[i].z);
// 			glutSolidSphere(spheres[i].w, 16, 16);
// 			glPopMatrix();
// 		}
	}
	glPopMatrix();
	//mmglPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void vpolygon::setResultData(unsigned int n)
{
	if (!outPos)
		outPos = new VEC3D[n];
	if (!outRot)
		outRot = new EPD[n];
}

void vpolygon::insertResultData(unsigned int i, VEC3D& p, EPD& r)
{
	outPos[i] = p;
	outRot[i] = r;
}