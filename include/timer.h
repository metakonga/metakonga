#ifndef TIMER_H
#define TIMER_H

#include <windows.h>

namespace parSIM
{
	class timer
	{
	private:
		bool Stopped;
		LARGE_INTEGER Freq;
		LARGE_INTEGER CounterIni,CounterEnd;

		LARGE_INTEGER GetElapsed(){ 
			LARGE_INTEGER dif; dif.QuadPart=(Stopped? CounterEnd.QuadPart-CounterIni.QuadPart: 0);
			return(dif);
		}

	public:
		timer(){ QueryPerformanceFrequency(&Freq); Reset(); }
		void Reset(){ Stopped=false; CounterIni.QuadPart=0; CounterEnd.QuadPart=0; }
		void Start(){ Stopped=false; QueryPerformanceCounter(&CounterIni); }
		void Stop(){ QueryPerformanceCounter(&CounterEnd); Stopped=true; }
		//-Returns time in miliseconds.
		float GetElapsedTimeF(){ return((float(GetElapsed().QuadPart)/**float(1000)*/)/float(Freq.QuadPart)); }
		double GetElapsedTimeD(){ return((double(GetElapsed().QuadPart)/**double(1000)*/)/double(Freq.QuadPart)); }
	};
}

#endif