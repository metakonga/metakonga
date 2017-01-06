#include "cu_event.h"

cu_event::cu_event()
	: nEvent(0)
{

}

cu_event::~cu_event()
{

}

void cu_event::addEvent(const eventType& _event)
{
	events.push_back(_event);
}