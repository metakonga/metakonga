#ifndef SIMULATION_H
#define SIMULATION_H

class simulation
{
public:
	enum deviceType{ CPU = 0, GPU };
	simulation();
	~simulation();

	static bool isCpu();
	static bool isGpu();
	static void setCPUDevice();
	static void setGPUDevice();
	static void setTimeStep(double _dt);
	static void setCurrentTime(double _ct);
	static void setStartTime(double _st);

	static double start_time;
	static double et;
	static double init_dt;
	static double dt;
	static double ctime;
	static unsigned int st;
	static unsigned int nstep;
	static deviceType dev;
};

#endif