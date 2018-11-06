#ifndef MBD_MODEL_H
#define MBD_MODEL_H

#include <QMap>
#include <QFile>
#include <QTextStream>
#include <QString>
#include "kinematicConstraint.h"
#include "drivingConstraint.h"
#include "rigidBody.h"
#include "artificialCoordinate.h"
#include "model.h"
#include "algebraMath.h"

class forceElement;
class axialRotationForce;
class springDamperModel;

class mbd_model : public model
{
public:
	mbd_model();
	mbd_model(QString _name);
	virtual ~mbd_model();

//	rigidBody* createRigidBody(QString _name);
	pointMass* createPointMass(
		QString _name, double mass, VEC3D piner, VEC3D siner, 
		VEC3D p, EPD ep = EPD(1.0, 0.0, 0.0, 0.0));
	pointMass* createPointMassWithGeometry(
		int gt, QString _name, double mass, VEC3D piner, VEC3D siner,
		VEC3D p, EPD ep = EPD(1.0, 0.0, 0.0, 0.0));
	void insertPointMass(pointMass* _pm);
	pointMass* Ground();
	kinematicConstraint* createKinematicConstraint(
		QString _name, kinematicConstraint::Type kt, 
		pointMass* i, VEC3D& spi, VEC3D& fi, VEC3D& gi, 
		pointMass* j, VEC3D& spj, VEC3D& fj, VEC3D& gj);
	kinematicConstraint* createKinematicConstraint(QTextStream& qts);
// 	kinematicConstraint* createCableConstraint(
// 		QString _name,
// 		pointMass* fi, VEC3D& fspi, pointMass* fj, VEC3D& fspj,
// 		pointMass* si, VEC3D& sspi, pointMass* sj, VEC3D& sspj);
// 	kinematicConstraint* createGearConstraint(
// 		QString _name, pointMass* i, kinematicConstraint* ik, 
// 		pointMass* j, kinematicConstraint* jk, double r);
// 	kinematicConstraint* createGearConstraint(
// 		QString _name, kinematicConstraint* ik, kinematicConstraint* jk,
// 		double cx, double cy, double cz, double r);
// 	kinematicConstraint* createGearConstraint(QTextStream& qts);
// 	kinematicConstraint* createCableConstraint(QTextStream& qts);
	springDamperModel* createSpringDamperElement(
		QString _name,
		pointMass* i, VEC3D& bLoc,
		pointMass* j, VEC3D& aLoc,
		double k, double c);
	springDamperModel* createSpringDamperElement(QTextStream& qts);
	axialRotationForce* createAxialRotationForce(
		QString _name, pointMass* i, pointMass* j, VEC3D loc, VEC3D u, double v);
	axialRotationForce* createAxialRotationForce(QTextStream& qts);
	artificialCoordinate* createArtificialCoordinate(QString _nm);
	drivingConstraint* createDrivingConstraint(
		QString _nm, kinematicConstraint* _kin, drivingConstraint::Type _tp,
		double iv, double cv);

	//contactPair* createContactPair(QString _nm, pointMass* ib, pointMass* jb);

	void set2D_Mode(bool b);
	bool mode2D();
	void setMBDModelName(QString _n) { mbd_model_name = _n; }
	//QString& modelPath() { return model_path; }
	double StartTimeForSimulation() { return start_time_simulation; }
	QString& modelName() { return mbd_model_name; }
	pointMass* PointMass(QString nm);
	kinematicConstraint* kinConstraint(QString s);
	QMap<QString, kinematicConstraint*>& kinConstraint() { return consts; }
	QMap<QString, pointMass*>& pointMasses() { return masses; }
	QMap<QString, forceElement*>& forceElements() { return forces; }
	//QMap<QString, cableConstraint*>& cableConstraints() { return cables; }
	//QMap<QString, gearConstraint*>& gearConstraints() { return gears; }
	QMap<QString, artificialCoordinate*>& artificialCoordinates() { return acoordinates; }
	QMap<QString, drivingConstraint*>& drivingConstraints() { return drivings; }

	void Open(QTextStream& qts);
	void Save(QTextStream& qts);

	void exportPointMassResultData2TXT();
	void exportReactionForceResultData2TXT();
	void loadPointMassResultDataFromTXT();
	void saveModel(QTextStream& qts);
	void runExpression(double ct, double dt);
	void updateCableInitLength();
	void updateAndInitializing();
	void setStartTimeForSimulation(double sts);
//	QMap<QString, v3epd_type> setStartingData(startingModel* stm);
	virtual void userDefineProcess(){};

	//void setPointMassDataFromStartingModel(QMap<int, resultStorage::pointMassResultData>& d);

protected:
	pointMass *ground;
	VEC3D grav;

	//QString model_path;
	QString mbd_model_name;
	bool is2D;
	
	double start_time_simulation;

	QMap<QString, QString> body_logs;
	QMap<QString, QString> other_logs;
	QMap<QString, pointMass*> masses;
	QMap<QString, kinematicConstraint*> consts;
	//QMap<QString, cableConstraint*> cables;
	//QMap<QString, gearConstraint*> gears;
	QMap<QString, forceElement*> forces;
	QMap<QString, artificialCoordinate*> acoordinates;
//	QMap<QString, contactPair*> cpairs;
	QMap<QString, drivingConstraint*> drivings;
};

#endif