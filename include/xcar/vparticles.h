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
	void draw_f(GLenum eMode, int wHeight, int protype, float z);
	bool define();
	bool define(double* p, unsigned int n);
	bool define_f(float* p, unsigned int n);

	void setParticlePosition(double* p, unsigned int n);
	void resizeMemory(double* p, unsigned int n);
	void resizeMemory_f(float* p, unsigned int n);
	void openResultFromFile(unsigned int idx);

	void upParticleScale(double v) { pscale += v; }
	void downParticleScale(double v) { pscale -= v; }

private:
	//unsigned int _compileProgram(const char *vsource, const char *fsource);
	void _drawPoints();
	void _drawPoints_f();

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

	float *buffer_f;
	float *color_buffer_f;
	float *pos_f;
	float *vel_f;
	float *force_f;
	float *color_f;

	bool isSetColor;

	float pscale;

	shaderProgram program;
};


#endif