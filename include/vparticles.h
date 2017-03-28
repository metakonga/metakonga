#ifndef VIEW_PARTICLES_H
#define VIEW_PARTICLES_H

#include "vglew.h"
#include "vcontroller.h"

class particle_system;


class vparticles : public vglew
{
public:

	vparticles();
	vparticles(particle_system* ps);
	~vparticles();

	void draw(GLenum eMode, int wHeight, int protype);
	bool define();
	bool define(VEC4D* p, unsigned int _n);

	void calcMaxForce();
	void resizeMemory();
	void settingSphParticles(unsigned int np, QString file);
	double* getPosition() { return pos; }
	void changeParticles(VEC4D_PTR _pos);
	QString& Name() { return name; }
	unsigned int Np() { return np; }
	QString& BaseGeometryText() { return baseGeometry; }
	void openResultFromFile(unsigned int idx);
	void openResultFromFile_SPH(unsigned int idx);
	void setResultFileList(QStringList& fl) { rList = fl; }
	particle_system* getParticleSystem() { return ps; }

	void upParticleScale(double v) { pscale += v; }
	void downParticleScale(double v) { pscale -= v; }

private:
	unsigned int createVBO(unsigned int size, double *bufferData = 0);
	unsigned int _compileProgram(const char *vsource, const char *fsource);
	void _drawPoints();

	double MPForce;
	//	float maxPressure;

	unsigned int m_posVBO;
	unsigned int m_colorVBO;
	unsigned int m_program;

	QString name;
	QString baseGeometry;
	QStringList rList;
	unsigned int np;
	double *buffer;
	double *color_buffer;
	double *pos;
	double *vel;
	double *force;
	double *color;

	bool isSphParticle;
	bool isSetColor;
	bool isglewinit;

	float pscale;

	particle_system *ps;
};


#endif