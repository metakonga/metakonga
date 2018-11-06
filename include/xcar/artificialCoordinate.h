#ifndef ARTIFICIALCOORDINATE_H
#define ARTIFICIALCOORDIANTE_H

#include <QString>

class artificialCoordinate
{
public:
	artificialCoordinate(QString _name);
	~artificialCoordinate();

	QString getName() { return name; }
	unsigned int getMatrixLocation() { return matrix_location; }
	void setMatrixLocation(unsigned int ml) { matrix_location = ml; }

private:
	QString name;
	int id;
	unsigned int matrix_location;
	static int count;
};

#endif