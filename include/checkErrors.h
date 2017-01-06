#ifndef CHECKERRORS_H
#define CHECKERRORS_H

enum errorCode
{
	ReturnNULL = 0,
	Success = 1,
	ErrorZeroValue = 2,
};

static const char* getErrorCodeEnum(errorCode error)
{
	switch(error){
	case Success:
		return "Success";
	case ErrorZeroValue:
		return "ErrorZeroValue";
	case ReturnNULL:
		return "ReturnNULL";
	}
	return "<unknown>";
}

template< typename T >
void mycheck(T result, char const *const func, const char *const file, int const line)
{
	if (result != 1)
	{
		fprintf(stderr, "Error at %s:%d code = %d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), getErrorCodeEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

template< typename T >
void xmlCheck(T result, char const *const func, const char *const file, int const line)
{
	if (result < 0)
	{
		fprintf(stderr, "Error at %s:%d code = %d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
		exit(EXIT_FAILURE);
	}
}

#define checkErrors(val)	mycheck ( (val), #val, __FILE__, __LINE__ )
#define xmlCheckErrors(val) xmlCheck ( (val), #val, __FILE__, __LINE__ )

#endif