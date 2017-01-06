#include "Log.h"
#include <iostream>


std::string parSIM::Log::outputFile;
std::ofstream parSIM::Log::outputStream;
parSIM::Log::MessageType parSIM::Log::logLevel = parSIM::Log::DebugInfo;

void parSIM::Log::receiver(const Log::Message& m)
{
	switch(m.type)
	{
	case Log::Info:
		std::cout << "> " << m.text << std::endl; break;
	case Log::Warning:
		std::cout << "? " << m.text << std::endl; break;
	case Log::Error:
		std::cout << "! " << m.text << std::endl; break;
	case Log::User:
		std::cout << "> " << m.text << std::endl; break;
	case Log::DebugInfo:
		std::cout << "  " << m.text << std::endl; break;
	default: break;
	}
}

void parSIM::Log::Send(MessageType type, const std::string& text)
{
	if( (int)type < (int)logLevel )
		return;

	Log::Message m;
	m.type = type;
	m.text = text;
	time_t t = time(NULL);
	localtime_s(&m.when, &t);

	receiver(m);

	if(outputStream.is_open()){
		char buffer[9];
		strftime(buffer, 9, "%X", &m.when);
		outputStream << buffer;

		switch(type)
		{
		case DebugInfo: outputStream << " | debug   | "; break;
		case Info:		outputStream << " | info    | "; break;
		case Warning:	outputStream << " | warning | "; break;
		case Error:		outputStream << " | ERROR   | "; break;
		default:		outputStream << " | user    | ";
		}
		outputStream << text << std::endl;
	}
}

void parSIM::Log::SetOutput(const std::string& filename)
{
	LogDebug("Setting output log file: " + filename);
	outputFile = filename;

	// close old one
	if(outputStream.is_open())
		outputStream.close();

	// create file
	outputStream.open(filename.c_str());
	if(!outputStream.is_open())
		Send(Error, "Cannot create/open '" + filename + "' for logging");
}

void parSIM::Log::SetLevel(parSIM::Log::MessageType level) 
{
	logLevel = level;
}


parSIM::Log::~Log()
{
	if(outputStream.is_open())
		outputStream.close();
}