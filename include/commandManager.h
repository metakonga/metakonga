#ifndef COMMANDMANAGER_H
#define COMMANDMANAGER_H

#include <QString>
#include <QStringList>

class commandManager
{
public:
	commandManager();
	~commandManager();

	int QnA(QString& q);
	QString AnQ(int c);
	QString AnQ(QString c, QString v);
	bool IsFinished() { is_finished; }
	QString& SuccessMessage() { return successMessage; }
	QString& FailureMessage() { return failureMessage; }
	QString getPassedCommand();
private:
	bool is_finished;
	int sz;
	int cidx;
	int cstep;
	QStringList sList;
	int step0(int c, QString s);
	int step1(int c, QString s);
	int step2(int c, QString s);
	int step3(int c, QString s);
	int current_log_index;
	QStringList logs;
	QString successMessage;
	QString failureMessage;
};

#endif