#ifndef ERRORS_H
#define ERRORS_H

#include <QTextStream>

class errors
{
public:
	enum error_type{ NO_ERROR, MBD_EXCEED_NR_ITERATION };

	errors();
	~errors();
	static void setError(error_type _e) { e = _e; }
	static void Error(QString &target);

private:
	static error_type e;
};

#endif