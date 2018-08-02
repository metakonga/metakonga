#ifndef VGLEW_H
#define VGLEW_H

#include <stdio.h>
#include <stdlib.h>

#ifndef QT_OPENGL_ES_2
#include <gl/glew.h>
//#include <gl/freeglut.h>
//#include <gl/glu.h>
#endif
//#include <QGLWidget>

static class vglew
{
public:
	vglew() 
		: isInitGlew(false)
	{
		glewInit();
		isInitGlew = true;
	}
	vglew(int argc, char** argv)
		: isInitGlew(false)
	{
		//glutInit(&argc, argv);  
		glewInit();
		isInitGlew = true;
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
	bool isInitGlew;
}g_vglew;

class shaderProgram
{
public:
	shaderProgram()
		: m_program(0)
	{}
	~shaderProgram()
	{
		deleteProgram();
	}

	unsigned int Program()
	{
		return m_program;
	}

	void deleteProgram()
	{
		if (m_program) glDeleteProgram(m_program); m_program = 0;
	}

	void compileProgram(const char *vsource, const char *fsource)
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
		m_program = program;
	}

	unsigned int m_program;
};

//bool vglew::isGlewInit = false;

#endif