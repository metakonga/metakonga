#ifndef LOG_H
#define LOG_H

#include <string>
#include <list>
#include <fstream>
#include <ctime>

namespace parSIM
{
	class Log
	{
	public:
		Log(){}
		~Log();

		enum MessageType
		{
			DebugInfo,
			Info,
			Warning,
			Error,
			User
		};

		struct Message
		{
			MessageType type;
			std::string text;
			tm when;
		};

		static void SetOutput(const std::string& filename);

		static const std::string& Output() { return outputFile; }

		static void Send(MessageType type, const std::string& text);

		static void receiver(const Message&);

		static void SetLevel(MessageType level);

	private:
		static std::string outputFile;  
		static std::ofstream outputStream;
		static MessageType logLevel;
		//static void (*receiver)(const Message&);
	};
#ifdef NDEBUG
	#define LogDebug(DESC)
#else
	#define LogDebug(DESC) Log::Send(Log::DebugInfo, DESC)
#endif
}

#endif