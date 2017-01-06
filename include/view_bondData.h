#ifndef VIEW_BONDDATA_H
#define VIEW_BONDDATA_H

#include "Object.h"

namespace parview
{
// 	class bondData : public Object
// 	{
// 		struct bond_data
// 		{
// 			bool broken;
// 			float length;
// 			vector3<float> sp;
// 			vector3<float> ep;
// 		};
// 
// 	public:
// 		bondData();
// 		virtual ~bondData();
// 
// 		void setBondData(QFile& pf);
// 		virtual void SaveObject(QTextStream& out) {}
// 		virtual bool callDialog(DIALOGTYPE dt = NEW_OBJECT) { return true; }
// 		virtual void draw(GLenum eMode);
// 		void define(void* tg = 0);
// 		virtual void saveCurrentData(QFile& pf);
// 		virtual void updateDataFromFile(QFile& pf, unsigned int fdtype){}
// 		virtual void hertzian_contact_force(void* p, void* v, void* w, void* f, void* m, float ma, float dt, parview::contactConstant* cc){}
// 		bond_data *bds;
// 		unsigned int size;
// 
// 		unsigned int glList;
// 
// 		private slots:
// 		virtual void Click_ok(){}
// 		virtual void Click_cancel(){}
// 	};
}

#endif