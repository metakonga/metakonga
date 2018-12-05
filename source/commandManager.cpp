#include "commandManager.h"
#include "modelManager.h"
#include "glwidget.h"
#include <QTextStream>

contact* cont = NULL;
pointMass* mass = NULL;

int commandManager::step0(int c, QString s)
{
	if (c == 1) // event
	{
		if (s == "mbd")
			if (sz > 2)
				return step1(11, sList.at(++cidx));
		else
			return 11;
		if (s == "contact")
			if(sz > 2)
				return step1(12, sList.at(++cidx));
		else
			return 12;
	}
	else if (c == 2) // RV
	{
		mbd_model* mbd = modelManager::MM()->MBDModel();
		if (mbd)
		{
			mass = mbd->PointMass(s);
			if (mass)
				if (sz == 5)
				{
					VEC3D rv(sList.at(++cidx).toDouble(), sList.at(++cidx).toDouble(), sList.at(++cidx).toDouble());
					mass->setRotationalVelocity(rv);
					//return step1(21, sList.at(++cidx));
				}					
				else
					return 21;
		}		
		return 20;
	}
	else if (c == 4) // env
	{
		if (s == "rolling")
		{
			if (sz > 2)
				return step1(41, sList.at(++cidx));
			else
			{
				if (modelManager::MM()->DEMModel())
				{
					bool b = modelManager::MM()->DEMModel()->RollingCondition();
					QTextStream(&successMessage) << "Rolling condition : " << (b ? "enable" : "disable");
					return 0;
				}
				QTextStream(&failureMessage) << "DEM model is not exist.";
			}
		}
	}
	else
		return c;
	return -1;
}

int commandManager::step1(int c, QString s)
{
	if (c == 11) // event -> mbd
	{
		if (s == "start_time")
			if(sz > 3)
				return step2(111, sList.at(++cidx));
		else
			return 111;
	}
	else if (c == 12) // event -> contact
	{
		contactManager* cm = modelManager::MM()->ContactManager();
		if (cm)
		{
			cont = cm->Contact(s);
			if (cont)
				if(sz > 3)
					return step2(121, sList.at(++cidx));
			else
				return 121;
		}
		return 120;
	}
	else if (c == 41) // env rolling
	{
		bool b = (bool)s.toInt();
		if (modelManager::MM()->DEMModel())
		{
			modelManager::MM()->DEMModel()->setRollingConditionEnable(b);
			QTextStream(&successMessage) << "Rolling condition of dem model is " << (b ? "enabled." : "disabled.");
			return 0;
		}
		QTextStream(&failureMessage) << "DEM model is not exist.";
	}
// 	else if (c == 21)
// 	{
// 		if (sList.size() != 5)
// 			return -1;
// 		VEC3D rv(s.toDouble(), sList.at(++cidx).toDouble(), sList.at(++cidx).toDouble());
// 		mass = 
// 	}
	return -1;
}

int commandManager::step2(int c, QString s)
{
	if (c == 111) // event -> mbd -> start_time
	{
		mbd_model* mbd = modelManager::MM()->MBDModel();
		if (mbd)
			mbd->setStartTimeForSimulation(s.toDouble());
		return 0;
	}
	else if (c == 121)
	{
		if (s == "ignore_time")
			if(sz > 4)
				return step3(1211, sList.at(++cidx));
		else
			return 1211;
	}
	return -1;
}

int commandManager::step3(int c, QString s)
{
	if (c == 1211)
	{
		if (cont)
			cont->setIgnoreTime(s.toDouble());
		cont = NULL;
		return 0;
	}
	
	return -1;
}

commandManager::commandManager()
	: is_finished(false)
	, sz(0)
	, cidx(0)
	, cstep(0)
	, current_log_index(0)
{

}

commandManager::~commandManager()
{

}

int commandManager::QnA(QString& q)
{
	logs.push_back(q);
	current_log_index = 0;
	sList = q.split(" ");
	sz = sList.size();
	QString a;
	QTextStream qts;
	int c = 0;
	cidx = 0;
	if (sList.at(cidx) == "event")
	{
		cstep = 1;
		cidx++;
		if (sz > 1)	return step0(1, sList.at(cidx));
		else return 1;
	}
	else if (sList.at(cidx) == "RV")
	{
		cidx++;
		if (sz > 1) return step0(2, sList.at(cidx));
		else return 2;
	}
	else if (sList.at(cidx) == "env")
	{
		cidx++;
		if (sz > 1) return step0(4, sList.at(cidx));
	}
	//else if (sList.at(cidx) == "")
// 	if (cstep != 1) return -2;
// 	// step 2
// 	if (sList.at(cidx) == "mbd")
// 	{
// 		cstep = 2;
// 		cidx++;
// 		if (sz > 1) return step1(11, sList.at(cidx));
// 		else return 11;
// 	}
// 	if (cstep != 2) return -2;
// 	// step 3
// 	if (sList.at(cidx) == "start_time")
// 	{
// 		cstep = 3;
// 		cidx++;
// 		if (sz > 1) return step2(111, sList.at(cidx));
// 		else return 111;
// 	}

	return -2;
}

QString commandManager::AnQ(int c)
{
	QString q = "you can select as follows\n";
	switch (c)
	{
	case 1:   q += "    mbd\ncontact\n"; break;
	case 11:  q += "    start_time (value)\n"; break;
	case 111: q += "    (value)\n"; break;
	case 12:  
		if (modelManager::MM()->ContactManager())
			foreach(QString s, modelManager::MM()->ContactManager()->Contacts().keys())
				q += "    " + s + "\n";
		break;
	case 120: q = "Contact manager is not created."; break;
	case 121: q += "	ignore_time (value)\n"; break;
	case 1211: q += "   (value)\n"; break;
	default:  q = "Incorrect code.\n"; break;
	}
	return q;
}

QString commandManager::AnQ(QString c, QString v)
{
	QString ret;
	//QString log;
	if (c == "Input the refinement size.")
	{
		vobject* obj = GLWidget::GLObject()->selectedObjectWithCast();
		ret = modelManager::MM()->GeometryObject()->polyRefinement(obj->name(), v.toDouble());
	}
	//logs.push_back(log);
	return ret;
}

QString commandManager::getPassedCommand()
{
	QString c;
	int sz_log = logs.size();
	if (!current_log_index)
		current_log_index = sz_log;
	if (sz_log)
	{
		c = logs.at(current_log_index - 1);
	}
	return c;
}
