// 
// #include "view_object.h"
// #include "view_controller.h"
// #include "view_mass.h"
// 
// using namespace parview;
// 
// object::object()
// {
// //	memset(position, 0, sizeof(vector3<double>)*MAX_FRAME);
// }
// 
// object::~object()
// {
// 
// }
// 
// void object::setObjectData(QFile& pf)
// {
// 	type = OBJECT;
// 	int name_size = 0;
// 	vector3<double> pos;
// 	char nm[256] = { 0, };
// 	pf.read((char*)&name_size, sizeof(int));
// 	pf.read((char*)nm, sizeof(char)*name_size);
// 	name.sprintf("%s", nm);
// 	unsigned int line_size = 0;
// 	pf.read((char*)&pos.x, sizeof(vector3<double>));
// 	position = vector3<float>((float)pos.x, (float)pos.y, (float)pos.z);
// 	pf.read((char*)&line_size, sizeof(unsigned int));
// 	lines.alloc(line_size);
// 	pf.read((char*)lines.get_ptr(), sizeof(sline)*line_size);
// 	unsigned int point_size = 0;
// 	pf.read((char*)&point_size, sizeof(unsigned int));
// 	points.alloc(point_size);
// 	pf.read((char*)points.get_ptr(), sizeof(vector3<double>)*point_size);
// }
// 
// void object::draw(GLenum eMode)
// {
// 	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
// 	glPushMatrix();
// 	float* p = NULL;
// 	if (ms){
// 		p = ms->Position(view_controller::getRealTimeParameter() ? 0 : view_controller::getFrame());
// 	}
// 	else{
// 		p = &(position.x);
// 	}
// 	glTranslatef(p[0], p[1], p[2]);
// 	glDisable(GL_LIGHTING);
// 	glColor3fv(Object::color);
// 	glCallList(glList);
// 	glPopMatrix();
// }
// 
// void object::define(void* tg)
// {
// 	glList = glGenLists(1);
// 	glNewList(glList, GL_COMPILE);
// 	glShadeModel(GL_SMOOTH);
// 	glColor3f(1.0f, 0.0f, 0.0f);
// 	if (lines.sizes()){
// 		for (unsigned int i = 0; i < lines.sizes(); i++){
// 			glBegin(GL_LINES);
// 			{
// 				glVertex3f(lines(i).sp.x - position.x, lines(i).sp.y - position.y, lines(i).sp.z - position.z);
// 				glVertex3f(lines(i).ep.x - position.x, lines(i).ep.y - position.y, lines(i).sp.z - position.z);
// 			}
// 		}
// 	}
// 	
// 	glEnd();
// 	glEndList();
// }
// 
// 
// void object::saveCurrentData(QFile& pf)
// {
// 
// }