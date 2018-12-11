#ifndef EVENT_TRIGGER_H
#define EVENT_TRIGGER_H

#include <QString>

class event_trigger
{
public:
	event_trigger();
	~event_trigger();

	static void TriggerEvent(QString _msg) { isTrigger = true; msg = _msg; }
	static bool IsEvnetTrigger() { return isTrigger; }
	static QString OnMessage() 
	{ 
		QString ret_msg = msg;
		msg = "";
		isTrigger = false; 
		return ret_msg;
	}

private:
	static bool isTrigger;
	static QString msg;
};



#endif