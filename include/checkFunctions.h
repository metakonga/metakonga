#ifndef CHECKFUNCTIONS_H
#define CHECKFUNCTIONS_H

#include <QStringList> 

inline bool checkParameter3(QString& str)
{
	QStringList strList = str.split(" ");
	if (strList.size() != 3){
		//le->setStyleSheet("QLineEdit{background: yellow};");
		return false;
	}
	else{
		//le->setStyleSheet("QLineEdit{background: white};");
		return true;
	}
}

#endif