#ifndef VIEW_CONTROLLER
#define VIEW_CONTROLLER

#define MAX_FRAME 1000


class vcontroller
{
public:
	vcontroller();
	~vcontroller();

	static bool is_end_frame();
	static void move2previous2x();
	static void move2previous1x();
	static void on_play();
	static bool Play();
	static void off_play();
	static void move2forward1x();
	static void move2forward2x();
	static unsigned int getTotalBuffers() { return buffer_count; }
	static void setFrame(unsigned int f) { current_frame = f; }
	static int getFrame() { return (int)current_frame; }
	static void upBufferCount() { buffer_count++; }
	static void addTimes(unsigned int i, float time) { times[i] = time; }
	static float getTimes() { return times[current_frame]; }
	static float& getTimes(unsigned int i) { return times[i]; }
	static void setTotalFrame(unsigned int tf) { buffer_count = tf; }
	static void setRealTimeParameter(bool rt) { real_time = rt; }
	static bool getRealTimeParameter() { return real_time; }

private:
	static unsigned int current_frame;
	static unsigned int buffer_count;
	static bool is_play;
	static bool real_time;
	static float *times;
};


#endif