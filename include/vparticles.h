#ifndef VIEW_PARTICLES_H
#define VIEW_PARTICLES_H

#include "vglew.h"
#include "vcontroller.h"

class vparticles : public vglew
{
public:

	vparticles();
	~vparticles();

	void draw(GLenum eMode, int wHeight, int protype, double z);
	bool define();
	bool define(double* p, unsigned int n);

	void resizeMemory(double* p, unsigned int n);
	void openResultFromFile(unsigned int idx);

	void upParticleScale(double v) { pscale += v; }
	void downParticleScale(double v) { pscale -= v; }

private:
	//unsigned int _compileProgram(const char *vsource, const char *fsource);
	void _drawPoints();

	bool isDefine;

	unsigned int m_posVBO;
	unsigned int m_colorVBO;
	unsigned int m_program;

	unsigned int np;
	double *buffer;
	double *color_buffer;
	double *pos;
	double *vel;
	double *force;
	double *color;

	bool isSetColor;

	float pscale;

	shaderProgram program;
};


#endif