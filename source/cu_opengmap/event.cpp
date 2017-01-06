#include "event.h"

event::event()
	: nEvent(0)
{

}

event::~event()
{

}

void event::addEvent(const eventType& _event)
{
	events.push_back(_event);
}