#ifndef VIEW_OBJECT_H
#define VIEW_OBJECT_H

#include "Object.h"
#include "algebra/vector.hpp"

typedef struct
{
	vector3<double> sp;
	vector3<double> ep;
	vector3<double> nor;
}sline;

namespace parview
{
// 	class object : public Object
// 	{
// 	public:
// 		object();
// 		virtual ~object();
// 
// 		void setObjectData(QFile& pf);
// 
// 		void draw_object();
// 		virtual bool callDialog(DIALOGTYPE dt = NEW_OBJECT) { return true; }
// 		virtual void draw(GLenum eMode);
// 		void define(void* tg = 0);
// 		virtual void SaveObject(QTextStream& out) {}
// 		virtual void saveCurrentData(QFile& pf);
// 		virtual void updateDataFromFile(QFile& pf, unsigned int fdtype){}
// 		virtual void hertzian_contact_force(void* p, void* v, void* w, void* f, void* m, float ma, float dt, parview::contactConstant* cc){}
// 
// 	private:
// 		algebra::vector<sline> lines;
// 		algebra::vector<algebra::vector3<double>> points;
// 
// 		vector3<float> position;
// 		vector3<float> velocity;
// 		//vector3<double> position[MAX_FRAME];
// 		GLint glList;
// 
// 		private slots:
// 		virtual void Click_ok(){}
// 		virtual void Click_cancel(){}
// 	};
}


#endif