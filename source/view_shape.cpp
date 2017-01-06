// #include "view_shape.h"
// #include "view_controller.h"
// #include "view_mass.h"
// #include <QTextStream>
// 
// using namespace parview;
// 
// shape::shape()
// 	: v_size(0)
// 	, i_size(0)
// 	, vertice(NULL)
// 	, indice(NULL)
// 	, normals(NULL)
// {
// 
// }
// 
// 
// shape::~shape()
// {
// 	if (vertice) delete[] vertice; vertice = NULL;
// 	if (indice) delete[] indice; indice = NULL;
// 	if (normals) delete[] normals; normals = NULL;
// }
// 
// void shape::setShapeData(QFile& pf, unsigned int fdtype)
// {
// 	if (fdtype == 4){
// 		type = SHAPE;
// 		int name_size = 0;
// 		char nm[256] = { 0, };
// 		pf.read((char*)&name_size, sizeof(int));
// 		pf.read((char*)nm, sizeof(char)*name_size);
// 		name.sprintf("%s", nm);
// 
// 		pf.read((char*)&v_size, sizeof(unsigned int));
// 		pf.read((char*)&i_size, sizeof(unsigned int));
// 
// 		vertice = new vector3<float>[v_size];
// 		normals = new float[i_size * 3];
// 		indice = new vector3<unsigned int>[i_size];
// 
// 		pf.read((char*)&position, sizeof(vector3<float>));
// 		pf.read((char*)vertice, sizeof(vector3<float>)*v_size);
// 		pf.read((char*)indice, sizeof(vector3<unsigned int>)*i_size);
// 
// 		for (unsigned int i = 0; i < i_size; i++){
// 			vector3<unsigned int> index = indice[i];
// 
// 			vector3<float> P = vertice[index.x];
// 			vector3<float> Q = vertice[index.y];
// 			vector3<float> R = vertice[index.z];
// 			vector3<float> V = Q - P;
// 
// 			vector3<float> W = R - P;
// 			vector3<float> N = V.cross(W);
// 			vector3<float> n = N / N.length();
// 			normals[i * 3 + 0] = n.x;
// 			normals[i * 3 + 1] = n.y;
// 			normals[i * 3 + 2] = n.z;
// 		}
// 	}
// 	else{
// 		type = SHAPE;
// 		int name_size = 0;
// 		char nm[256] = { 0, };
// 		pf.read((char*)&name_size, sizeof(int));
// 		pf.read((char*)nm, sizeof(char)*name_size);
// 		name.sprintf("%s", nm);
// 
// 		pf.read((char*)&v_size, sizeof(unsigned int));
// 		pf.read((char*)&i_size, sizeof(unsigned int));
// 		vector3<double>* vert = new vector3<double>[v_size];
// 		vector3<double> pos;
// 		pf.read((char*)&pos, sizeof(vector3<double>));
// 		pf.read((char*)vert, sizeof(vector3<double>)*v_size);
// 		position = vector3<float>(static_cast<float>(pos.x), static_cast<float>(pos.y), static_cast<float>(pos.z));
// 
// 		vertice = new vector3<float>[v_size];
// 		normals = new float[i_size * 3];
// 		indice = new vector3<unsigned int>[i_size];
// 
// 		pf.read((char*)indice, sizeof(vector3<unsigned int>)*i_size);
// 
// 		for (unsigned int i = 0; i < v_size; i++){
// 			vertice[i] = vector3<float>(
// 				static_cast<float>(vert[i].x),
// 				static_cast<float>(vert[i].y),
// 				static_cast<float>(vert[i].z));
// 		}
// 
// 		for (unsigned int i = 0; i < i_size; i++){
// 			vector3<unsigned int> index = indice[i];
// 
// 			vector3<float> P = vertice[index.x];
// 			vector3<float> Q = vertice[index.y];
// 			vector3<float> R = vertice[index.z];
// 			vector3<float> V = Q - P;
// 
// 			vector3<float> W = R - P;
// 			vector3<float> N = V.cross(W);
// 			vector3<float> n = N / N.length();
// 			normals[i * 3 + 0] = n.x;
// 			normals[i * 3 + 1] = n.y;
// 			normals[i * 3 + 2] = n.z;
// 		}
// 
// 	}
// 	
// }
// 
// void shape::draw(GLenum eMode)
// {
// 	glPushMatrix();
// 	float* p = NULL;
// 	if (ms){
// 		p = ms->Position(view_controller::getRealTimeParameter() ? 0 : view_controller::getFrame());
// 	}
// 	else{
// 		p = &(position.x);
// 	}
// 	glTranslatef(p[0], p[1], p[2]);
// 	glRotatef(90, 0, 0, 1);
// 	glRotatef(-30, 1, 0, 0);
// 	glRotatef(-90, 0, 0, 1);
// 	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
// 	glEnable(GL_LIGHTING);
// 	glEnable(GL_COLOR_MATERIAL);
// 	for (unsigned int i = 0; i < i_size; i++){
// 		glBegin(GL_TRIANGLES);
// 		{
// 			glNormal3fv(&normals[i * 3 + 0]);
// 			glVertex3fv(&vertice[indice[i].x].x);
// 			glVertex3fv(&vertice[indice[i].y].x);
// 			glVertex3fv(&vertice[indice[i].z].x);
// 		}
// 		glEnd();
// 	}
// 
// 	glDisable(GL_LIGHTING);
// 	glPopMatrix();
// }
// 
// void shape::draw_shape()
// {
// 	glEnableClientState(GL_NORMAL_ARRAY);
// 	glEnableClientState(GL_VERTEX_ARRAY);
// 
// 	glBindBuffer(GL_ARRAY_BUFFER, vertice_id);
// 	glVertexPointer(3, GL_FLOAT, 0, 0);
// 
// 	glBindBuffer(GL_ARRAY_BUFFER, normals_id);
// 	glNormalPointer(GL_FLOAT, 0, 0);
// 
// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indice_id);
// 	glIndexPointer(GL_UNSIGNED_INT, 0, 0);
// 
// 	//glDrawArrays(GL_TRIANGLES, 0, i_size*3*);
// 	glDrawElements(GL_TRIANGLES, 3*i_size, GL_UNSIGNED_INT, (GLuint*)0 + 0);
// 
// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
// 
// 	glDisableClientState(GL_VERTEX_ARRAY);
// 	glDisableClientState(GL_NORMAL_ARRAY);
// }
// 
// void shape::define(void* tg)
// {
// 	glewInit();
// 
// 	vertice_id = 0;
// 	normals_id = 0;
// 	indice_id = 0;
// 	glGenBuffers(1, &vertice_id);
// 	glGenBuffers(1, &normals_id);
// 	glGenBuffers(1, &indice_id);
// 
// 	// vertice
// 	glBindBuffer(GL_ARRAY_BUFFER, vertice_id);
// 	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * v_size, vertice, GL_DYNAMIC_DRAW);
// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
// 
// 	// normals
// 	glBindBuffer(GL_ARRAY_BUFFER, normals_id);
// 	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * i_size, normals, GL_DYNAMIC_DRAW);
// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
// 
// 	// indice
// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indice_id);
// 	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 3 * i_size, indice, GL_STATIC_DRAW);
// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
// }
// 
// void shape::saveCurrentData(QFile& pf)
// {
// 	if (!ms)
// 		return;
// 
// 	pf.write((char*)&type, sizeof(int));
// 	unsigned int name_size = name.size();
// 	pf.write((char*)&name_size, sizeof(unsigned int));
// 	pf.write((char*)name.toStdString().c_str(), sizeof(char) * name_size);
// 	if (view_controller::getRealTimeParameter()){
// 		pf.write((char*)ms->Position(0), sizeof(float) * 3);
// 		pf.write((char*)ms->Velocity(0), sizeof(float) * 3);
// 		return;
// 	}
// 	float* cpos = ms->Position(view_controller::getFrame());
// 	float* cvel = ms->Velocity(view_controller::getFrame());
// 	pf.write((char*)cpos, sizeof(float) * 3);
// 	pf.write((char*)cvel, sizeof(float) * 3);
// }
// 
// void shape::updateDataFromFile(QFile& pf, unsigned int fdtype)
// {
// 	if (vertice) { delete[] vertice; vertice = NULL; }
// 	if (indice) { delete[] indice; indice = NULL; }
// 	if (normals) { delete[] normals; normals = NULL; }
// 	QString str;
// 	QTextStream inf(&pf);
// 	algebra::vector<vector3<float>> l_vertice;
// 	algebra::vector<vector3<unsigned int>> ids;
// 	vector3<float> vertex;
// 	vector3<float> l_vertex;
// 	vector3<unsigned int> index;
// 	while (!pf.atEnd()){
// 		inf >> str;
// 		if (str == "GRID*")
// 		{
// 			inf >> str >> vertex.x >> vertex.z >> str >> str >> str >> vertex.y;
// 			vertex = vertex / 1000.0;
// 			l_vertex = vertex;
// 			l_vertice.push(l_vertex);
// 		}
// 		if (str == "CTRIA3"){
// 			inf >> str >> str >> index.x >> index.y >> index.z;
// 			ids.push(index - vector3<unsigned int>(1,1,1));
// 		}
// 	}
// 
// 	l_vertice.adjustment();
// 	ids.adjustment();
// 
// 	v_size = l_vertice.sizes();
// 	i_size = ids.sizes();
// 
// 	normals = new float[i_size * 3];
// 	vertice = new vector3<float>[l_vertice.sizes()];
// 	indice = new vector3<unsigned int>[ids.sizes()];
// 
// 	memcpy(vertice, l_vertice.get_ptr(), sizeof(vector3<float>)*v_size);
// 	memcpy(indice, ids.get_ptr(), sizeof(vector3<unsigned int>)*i_size);
// 
// 	for (unsigned int i = 0; i < i_size; i++){
// 		vector3<unsigned int> index = indice[i];
// 
// 		vector3<float> P = vertice[index.x];
// 		vector3<float> Q = vertice[index.y];
// 		vector3<float> R = vertice[index.z];
// 		vector3<float> V = Q - P;
// 
// 		vector3<float> W = R - P;
// 		vector3<float> N = V.cross(W);
// 		vector3<float> n = N / N.length();
// 
// 		normals[i * 3 + 0] = n.x;
// 		normals[i * 3 + 1] = n.y;
// 		normals[i * 3 + 2] = n.z;
// 	}
// }