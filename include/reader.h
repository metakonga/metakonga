#ifndef READER_H
#define READER_H

#include "file_system.h"

namespace parSIM
{
	class reader : public file_system
	{
	public:
		reader();
		virtual ~reader(){};

		virtual bool run();
	};
}

#endif 