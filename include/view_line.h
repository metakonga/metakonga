#ifndef VIEW_LINE_H
#define VIEW_LINE_H

#include "vobject.h"
#include <QDialog>

QT_BEGIN_NAMESPACE
class QLineEdit;
class QPushButton;
QT_END_NAMESPACE

namespace parview
{
	class line// : public Object
	{
	public:
		line();
		virtual ~line();
// 
// 		void setLineData(QFile& pf);
// 		virtual bool callDialog(DIALOGTYPE dt = NEW_OBJECT);
// 		virtual void draw(GLenum eMode);
// 		virtual void SaveObject(QTextStream& out);
// 		bool define(void* tg = 0);
// 		virtual void saveCurrentData(QFile& pf);
// 		virtual void updateDataFromFile(QFile& pf, unsigned int fdtype);
// 		virtual void hertzian_contact_force(void* p, void* v, void* w, void* f, void* m, float ma, float dt, parview::contactConstant* cc){}
// 		void SetDataFromFile(QTextStream& in);
// 		float startPoint[3];
// 		float endPoint[3];
// 
// 		unsigned int glList;
// 
// 		QLineEdit *LEPa;
// 		QLineEdit *LEPb;
// 
// 		QDialog *lineDialog;
// 		static unsigned int nline;
// 		private slots:
// 		virtual void Click_ok();
// 		virtual void Click_cancel();
	};
}

#endif