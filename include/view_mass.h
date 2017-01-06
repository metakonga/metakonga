#ifndef VIEW_MASS
#define VIEW_MASS

#include "Object.h"


namespace parview
{
// 	class mass : public Object
// 	{
// 	public:
// 		mass();
// 		virtual ~mass();
// 
// 		void setMassData(QFile& pf);
// 		float* Position(unsigned int i) { return &pos[i].x; }
// 		float* Velocity(unsigned int i) { return &vel[i].x; }
// 		vector3<float>& Force(unsigned int i) { return force[i]; }
// 		virtual bool callDialog(DIALOGTYPE dt = NEW_OBJECT) { return true; }
// 		virtual void draw(GLenum eMode);
// 		virtual void SaveObject(QTextStream& out) {}
// 		void define(void* tg = 0);
// 		virtual void saveCurrentData(QFile& pf);
// 		virtual void updateDataFromFile(QFile& pf, unsigned int fdtype);
// 		virtual void hertzian_contact_force(void* p, void* v, void* w, void* f, void* m, float ma, float dt, parview::contactConstant* cc){}
// 
// 	private:
// 		vector3<float> pos[MAX_FRAME];
// 		vector3<float> vel[MAX_FRAME];
// 		vector3<float> force[MAX_FRAME];
// 
// 		private slots:
// 		virtual void Click_ok(){}
// 		virtual void Click_cancel(){}
//	};
}

#endif