#ifndef VGLEW_H
#define VGLEW_H

#ifndef QT_OPENGL_ES_2
#include <gl/glew.h>
//#include <gl/freeglut.h>
#include <gl/glu.h>
#endif

#include "mphysics_types.h"
#include "mphysics_numeric.h"

static class vglew
{
public:
	vglew() { glewInit(); }
	vglew(int argc, char** argv) {
		//glutInit(&argc, argv);  
		glewInit();
	}
	~vglew() {}

	template<typename T>
	unsigned int createVBO(unsigned int size, T *bufferData)
	{
		GLuint vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, size, bufferData, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		return vbo;
	}
}g_vglew;

//bool vglew::isGlewInit = false;

#endif