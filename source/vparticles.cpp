#include "vparticles.h"
#include "shader.h"
#include "colors.h"
#include "msgBox.h"
#include "particle_system.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <QStringList>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QGridLayout>
#include <QComboBox>
#include <QTabWidget>
#include <QDialogButtonBox>

#include "vcube.h"

vparticles::vparticles()
//: vglew()
	: np(0)
	//, maxPressure(15374.f)
	, isSetColor(false)
	, name("particles")
	, isglewinit(false)
	, pos(NULL)
	, vel(NULL)
	, force(NULL)
	, color(NULL)
	, isSphParticle(false)
	, pscale(0)
{
	m_posVBO = 0;
	m_colorVBO = 0;
	m_program = 0;
}

vparticles::vparticles(particle_system* _ps)
	//: vglew()
	: ps(_ps)
	, name(_ps->name())
	, isSetColor(false)
	, np(0)
	, isglewinit(false)
	, pos(NULL)
	, color(NULL)
	, vel(NULL)
	, force(NULL)
	, isSphParticle(false)
	, pscale(0)
{
	m_posVBO = 0;
	m_colorVBO = 0;
	m_program = 0;
}

vparticles::~vparticles()
{
	for (int i = 0; i < MAX_FRAME; i++){
		if (pos) delete[] pos; pos = NULL;
		if (color) delete[] color; color = NULL;
	}
	if (vel) delete[] vel; vel = NULL;
	if (force) delete[] force; force = NULL;
	if (m_posVBO){
		glDeleteBuffers(1, &m_posVBO);
		m_posVBO = 0;
	}
	if (m_colorVBO){
		glDeleteBuffers(1, &m_colorVBO);
		m_colorVBO = 0;
	}
	if (m_program){ glDeleteProgram(m_program); m_program = 0; }
}

void vparticles::settingSphParticles(unsigned int _np, QString file)
{
	np = _np;
	//np = ps->numParticle();
	char v;
	pos = new double[np * 4];
	color = new double[np * 4];
	//ps->setPosition(pos);
	QFile pf(file);
	float* v3 = new float[3];
	float pressure = 0.0;
	bool isFS = false;
	pf.open(QIODevice::ReadOnly);
	double maxPressure = 1800.0;
	bool* fs = new bool[np];
	char* ptype = new char[np];
	double* presses = new double[np];
	for (unsigned int i = 0; i < np; i++){
		pf.read((char*)&v, sizeof(char));
		ptype[i] = v;
		pf.read((char*)v3, sizeof(float) * 3);
		pos[i * 4 + 0] = (double)v3[0];
		pos[i * 4 + 1] = (double)v3[1];
		pos[i * 4 + 2] = (double)v3[2];
		pos[i * 4 + 3] = 0.0005f;
		if (v == 'f')
		{
			color[i * 4 + 0] = 0.0f;
			color[i * 4 + 1] = 0.0f;
			color[i * 4 + 2] = 1.0f;
			color[i * 4 + 3] = 1.0f;
		}		
		else if (v == 'b')
		{
			color[i * 4 + 0] = 1.0f;
			color[i * 4 + 1] = 0.0f;
			color[i * 4 + 2] = 0.0f;
			color[i * 4 + 3] = 1.0f;
		}
		else if (v == 'd')
		{
			color[i * 4 + 0] = 0.0f;
			color[i * 4 + 1] = 1.0f;
			color[i * 4 + 2] = 0.0f;
			color[i * 4 + 3] = 1.0f;
		}
		pf.read((char*)v3, sizeof(float) * 3);
		pf.read((char*)&pressure, sizeof(float));
		presses[i] = (double)pressure;
		if (maxPressure < pressure)
			maxPressure = pressure;
		pf.read((char*)&isFS, sizeof(bool));
		fs[i] = isFS;
// 		if (isFS)
// 		{
// 			color[i * 4 + 0] = 1.0f;
// 			color[i * 4 + 1] = 1.0f;
// 			color[i * 4 + 2] = 1.0f;
// 			color[i * 4 + 3] = 1.0f;
// 		}
	}
	double grad = maxPressure * 0.1f;
	double t = 0.f;
	for (unsigned int i = 0; i < np; i++){
		if (fs[i])
		{
			color[i * 4 + 0] = 1.0f;
			color[i * 4 + 1] = 1.0f;
			color[i * 4 + 2] = 1.0f;
			color[i * 4 + 3] = 1.0f;
			continue;
		}
		switch (ptype[i]){
		case 'f':
			t = (presses[i] - 0) / grad;
			ucolors::colorRamp(t, &(color[i * 4]));
			color[i * 4 + 3] = 1.0f;
			break;
		}
	}
	delete[] fs;
	delete[] v3;
	delete[] ptype;
	delete[] presses;
	pf.close();
	if (!np){
		msgBox("Particle generation is failed.", QMessageBox::Critical);
		return;
	}
// 	if (!isglewinit)
// 		glewInit();

	if (m_posVBO){
		glDeleteBuffers(1, &m_posVBO);
		m_posVBO = 0;
	}
	if (m_colorVBO){
		glDeleteBuffers(1, &m_colorVBO);
		m_colorVBO = 0;
	}
	unsigned int memSize = sizeof(double) * 4 * np;
	buffer = pos;
	color_buffer = color;
	if (!m_posVBO)
		m_posVBO = vglew::createVBO<double>(memSize, buffer);
	if (!m_colorVBO)
		m_colorVBO = vglew::createVBO<double>(memSize, color_buffer);

	if (!m_program)
		m_program = _compileProgram(vertexShader, spherePixelShader);

	isSphParticle = true;
	//return true;
}

void vparticles::openResultFromFile_SPH(unsigned int idx)
{
	char v;
	double* v3 = new double[3];
	double pressure = 0.f;
	bool isFS = false;
	double maxPressure = 0.f;
	bool* fs = new bool[np];
	char* ptype = new char[np];
	double* presses = new double[np];
	QFile pf(rList.at(vcontroller::getFrame()));
	pf.open(QIODevice::ReadOnly);
	for (unsigned int i = 0; i < np; i++){
		pf.read((char*)&v, sizeof(char));
		ptype[i] = v;
		pf.read((char*)v3, sizeof(double) * 3);
		pos[i * 4 + 0] = v3[0];
		pos[i * 4 + 1] = v3[1];
		pos[i * 4 + 2] = v3[2];
		pos[i * 4 + 3] = 0.00125;
		if (v == 'f')
		{
			color[i * 4 + 0] = 0.0f;
			color[i * 4 + 1] = 0.0f;
			color[i * 4 + 2] = 1.0f;
			color[i * 4 + 3] = 1.0f;
		}
		else if (v == 'b')
		{
			color[i * 4 + 0] = 1.0f;
			color[i * 4 + 1] = 0.0f;
			color[i * 4 + 2] = 0.0f;
			color[i * 4 + 3] = 1.0f;
		}
		else if (v == 'd')
		{
			color[i * 4 + 0] = 0.0f;
			color[i * 4 + 1] = 1.0f;
			color[i * 4 + 2] = 0.0f;
			color[i * 4 + 3] = 1.0f;
		}
		pf.read((char*)v3, sizeof(double) * 3);
		pf.read((char*)&pressure, sizeof(double));
		presses[i] = pressure;
		if (maxPressure < pressure)
			maxPressure = pressure;
		pf.read((char*)&isFS, sizeof(bool));
		fs[i] = isFS;
// 		if (isFS)
// 		{
// 			
// 			color[i * 4 + 0] = 1.0f;
// 			color[i * 4 + 1] = 1.0f;
// 			color[i * 4 + 2] = 1.0f;
// 			color[i * 4 + 3] = 1.0f;
// 		}
	}
	double grad = maxPressure * 0.1f;
	double t = 0.f;
	for (unsigned int i = 0; i < np; i++){
		/*if (maxPressure < )*/
		if (fs[i])
		{
			color[i * 4 + 0] = 1.0f;
			color[i * 4 + 1] = 1.0f;
			color[i * 4 + 2] = 1.0f;
			color[i * 4 + 3] = 1.0f;
			continue;
		}
// 		switch (ptype[i]){
// 		case 'f':
			t = (presses[i] - 0) / grad;
			ucolors::colorRamp(t, &(color[i * 4]));
			color[i * 4 + 3] = 1.0f;
			//break;
		//}
	}
	delete[] v3;
	delete[] fs;
	delete[] ptype;
	delete[] presses;
	pf.close();
}

void vparticles::draw(GLenum eModem, int wHeight, int protype)
{
	if (rList.size()){
		if(isSphParticle)
			this->openResultFromFile_SPH(vcontroller::getFrame());
		else
			this->openResultFromFile(vcontroller::getFrame());
		buffer = pos;
		color_buffer = color;
	}
	
	glDisable(GL_LIGHTING);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	
	glUseProgram(m_program);
	float projFactor = 0.f;
	if (!protype){
		float fTmp1[16] = { 0.f, };
		glGetFloatv(GL_MODELVIEW_MATRIX, fTmp1);
		pscale = wHeight*1.74;
		glUniform1f(glGetUniformLocation(m_program, "orthoDist"), abs(fTmp1[14]));
		projFactor = 10.f;
	}
	else{
		pscale = (wHeight) / tanf(60 * 0.5*M_PI / 180.0);		
	}
	//float ptype = 100.f;
	glUniform1f(glGetUniformLocation(m_program, "isOrgho"), projFactor);
	//glUniform1f(glGetUniformLocation(m_program, "pointScale"), (wHeight) / tanf(60 * 0.5*M_PI / 180.0));
	glUniform1f(glGetUniformLocation(m_program, "pointScale"), pscale);
	//glUniform1f(glGetUniformLocation(m_program, "orthoDist"), abs(fTmp1[14]));

	_drawPoints();

	glUseProgram(0);
	glDisable(GL_POINT_SPRITE_ARB);
	glEnable(GL_LIGHTING);
}

void vparticles::_drawPoints()
{
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	if (m_posVBO)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_posVBO);
		glVertexPointer(4, GL_DOUBLE, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLdouble)*np * 4, buffer);
		if (m_colorVBO)
		{
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
			glColorPointer(4, GL_DOUBLE, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLdouble)*np * 4, color_buffer);
			
		}

		glDrawArrays(GL_POINTS, 0, np);

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
}

bool vparticles::define()
{
	np = ps->numParticle();
	if (!pos) pos = new double[np * 4];
	if (!vel) vel = new double[np * 3];
	if (!force) force = new double[np * 3];
	if (!color) color = new double[np * 4];
	ps->setPosition(pos);
	for (unsigned int i = 0; i < np; i++){
		color[i * 4 + 0] = 0.0f;
		color[i * 4 + 1] = 0.0f;
		color[i * 4 + 2] = 1.0f;
		color[i * 4 + 3] = 1.0f;
	}
	if (!np){
		msgBox("Particle generation is failed.", QMessageBox::Critical);
		return false;
	}
// 	if (!isglewinit)
// 		glewInit();

	if (m_posVBO){
		glDeleteBuffers(1, &m_posVBO);
		m_posVBO = 0;
	}
	if (m_colorVBO){
		glDeleteBuffers(1, &m_colorVBO);
		m_colorVBO = 0;
	}
	unsigned int memSize = sizeof(double) * 4 * np;
	buffer = pos;
	color_buffer = color;
	if (!m_posVBO) 
		m_posVBO = vglew::createVBO<double>(memSize, buffer);
	if (!m_colorVBO)
		m_colorVBO = vglew::createVBO<double>(memSize, color_buffer);

	if (!m_program)
		m_program = _compileProgram(vertexShader, spherePixelShader);

	return true;
}

void vparticles::resizeMemory()
{
	double* tv4 = new double[np * 4];
	double* tv3 = new double[np * 3];
	unsigned int new_np = ps->numParticle();
	memcpy(tv4, pos, sizeof(double) * np * 4);
	delete[] pos;
	pos = new double[new_np * 4]; memcpy(pos, tv4, sizeof(double) * np * 4);
	memcpy(tv3, vel, sizeof(double) * np * 3); delete[] vel; vel = new double[new_np * 3]; memcpy(vel, tv3, sizeof(double) * np * 3);
	memcpy(tv3, force, sizeof(double) * np * 3); delete[] force; force = new double[new_np * 3]; memcpy(force, tv3, sizeof(double) * np * 3);
	memcpy(tv4, color, sizeof(double) * np * 4); delete[] color; color = new double[new_np * 4]; memcpy(color, tv4, sizeof(double) * np * 4);
	delete[] tv4;
	delete[] tv3;
	np = new_np;
}

bool vparticles::define(VEC4D* p, unsigned int _n)
{
	np = _n;// ps->numParticle();
	if(!pos) pos = new double[np * 4];
	if(!vel) vel = new double[np * 3];
	if(!force) force = new double[np * 3];
	if(!color) color = new double[np * 4];
	ps->setPosition(pos);
	for (unsigned int i = 0; i < np; i++){
		color[i * 4 + 0] = 0.0f;
		color[i * 4 + 1] = 0.0f;
		color[i * 4 + 2] = 1.0f;
		color[i * 4 + 3] = 1.0f;
	}
	if (!np){
		msgBox("Particle generation is failed.", QMessageBox::Critical);
		return false;
	}
	// 	if (!isglewinit)
	// 		glewInit();

	if (m_posVBO){
		glDeleteBuffers(1, &m_posVBO);
		m_posVBO = 0;
	}
	if (m_colorVBO){
		glDeleteBuffers(1, &m_colorVBO);
		m_colorVBO = 0;
	}
	unsigned int memSize = sizeof(double) * 4 * np;
	buffer = pos;
	color_buffer = color;
	if (!m_posVBO)
		m_posVBO = vglew::createVBO<double>(memSize, buffer);
	if (!m_colorVBO)
		m_colorVBO = vglew::createVBO<double>(memSize, color_buffer);

	if (!m_program)
		m_program = _compileProgram(vertexShader, spherePixelShader);

	return true;
}

// unsigned int vparticles::createVBO(unsigned int size, float *bufferData)
// {
// 	GLuint vbo;
// 	glGenBuffers(1, &vbo);
// 	glBindBuffer(GL_ARRAY_BUFFER, vbo);
// 	glBufferData(GL_ARRAY_BUFFER, size, bufferData, GL_DYNAMIC_DRAW);
// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
// 	
// 	return vbo;
// }

unsigned int vparticles::_compileProgram(const char *vsource, const char *fsource)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vsource, 0);
	glShaderSource(fragmentShader, 1, &fsource, 0);

	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success) {
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;
}

void vparticles::openResultFromFile(unsigned int idx)
{
	QFile pf(rList.at(vcontroller::getFrame()));
	pf.open(QIODevice::ReadOnly);
	double time = 0.f;
	unsigned int _np = 0;
	pf.read((char*)&_np, sizeof(unsigned int));
	pf.read((char*)&time, sizeof(double));
	pf.read((char*)pos, sizeof(double) * 4 * np);
	pf.read((char*)vel, sizeof(double) * 3 * np);
	pf.read((char*)force, sizeof(double) * 3 * np);

// 	double grad = MPForce * 0.1f;
// 	double t = 0.f;
// 	for (unsigned int i = 0; i < np; i++){
// 		double m = VEC3F(force[i * 3 + 0], force[i * 3 + 1], force[i * 3 + 2]).length();
// 		t = (m - 0) / grad;
// 		if (t > 7)
// 			m = m;
// 		ucolors::colorRamp(t, &(color[i * 4]));
// 		color[i * 4 + 3] = 1.0f;
// 		//break;
// 		//}
// 	}
// 	color[203878 * 4 + 0] = 1.0f;
// 	color[203878 * 4 + 1] = 1.0f;
// 	color[203878 * 4 + 2] = 1.0f;
	pf.close();
}

void vparticles::changeParticles(VEC4D_PTR _pos)
{
	memcpy(pos, _pos, sizeof(double) * 4 * np);
}

void vparticles::calcMaxForce()
{
	MPForce = 0.f;
	double *v4 = new double[np * 4];
	double *v3 = new double[np * 3];
	for (unsigned int i = 0; i < rList.size(); i++){
		QFile pf(rList.at(i));
		pf.open(QIODevice::ReadOnly);
		double time = 0.f;
		unsigned int _np = 0;

		pf.read((char*)&_np, sizeof(unsigned int));
		pf.read((char*)&time, sizeof(double));
		pf.read((char*)v4, sizeof(double) * 4 * np);
		pf.read((char*)v3, sizeof(double) * 3 * np);
		pf.read((char*)v3, sizeof(double) * 3 * np);
		for (unsigned int i = 0; i < np; i++){
			double m = sqrt(v3[i * 3 + 0] * v3[i * 3 + 0] + v3[i * 3 + 1] * v3[i * 3 + 1] + v3[i * 3 + 2] * v3[i * 3 + 2]);
			if (MPForce < m)
				MPForce = m;
		}
	}
	delete[] v4; v4 = NULL;
	delete[] v3; v3 = NULL;
}