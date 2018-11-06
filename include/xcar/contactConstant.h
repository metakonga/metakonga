#ifndef CONTACTCONSTANT_H
#define CONTACTCONSTANT_H

/*#include "Object.h"*/
#include "types.h"
#include "vectorTypes.h"
#include <QObject>
#include <QDialog>

QT_BEGIN_NAMESPACE
class QComboBox;
class QLineEdit;
class QTextStream;
QT_END_NAMESPACE

namespace parview
{
	class Object;
	class contactConstant : public QObject
	{
		Q_OBJECT

	public:
		contactConstant() 
			: obj_i(0)
			, obj_j(0)
			, restitution(0)
			, friction(0)
			, stiff_ratio(0)
			, ccdialog(NULL)
			, isDialogOk(false)
		{}
		contactConstant(const contactConstant &cc)
			: obj_si(cc.obj_si)
			, obj_sj(cc.obj_sj)
			, obj_i(cc.obj_i)
			, obj_j(cc.obj_j)
			, restitution(cc.restitution)
			, friction(cc.friction)
			, stiff_ratio(cc.stiff_ratio)
		{}
		~contactConstant()
		{
			//if (ccdialog) delete ccdialog; ccdialog = NULL;
// 			if (list1) delete list1; list1 = NULL;
// 			if (list2) delete list2; list2 = NULL;
// 			if (LErest) delete LErest; LErest = NULL;
// 			if (LEfric) delete LEfric; LEfric = NULL;
// 			if (LEratio) delete LEratio; LEratio = NULL;
		}
	
		//contact_coefficient_t CalcContactCoefficient(float ir, float jr, float im, float jm);
		bool callDialog(QStringList& strList);
		void SaveConstant(QTextStream& out);
		void SetDataFromFile(QTextStream& in);
		
		bool isDialogOk;

		QDialog *ccdialog;
		QComboBox *list1;
		QComboBox *list2;
		QLineEdit *LErest;
		QLineEdit *LEfric;
		QLineEdit *LEratio;

		QString obj_si;
		QString obj_sj;
		Object* obj_i;
		Object* obj_j;

		double restitution;
		double friction;
		double stiff_ratio;

		private slots:
		void clickOk();
		void clickCancel();
	};
}

#endif