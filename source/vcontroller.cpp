#include "vcontroller.h"
#include "Object.h"

unsigned int vcontroller::current_frame = 0;
unsigned int vcontroller::buffer_count = 0;
bool vcontroller::is_play = false;
float *vcontroller::times = new float[1000];
bool vcontroller::real_time = false;

vcontroller::vcontroller()
{

}

vcontroller::~vcontroller()
{
	if (times) delete[] times; times = NULL;
}

bool vcontroller::is_end_frame()
{
	if (is_play){
		/*current_frame++;*/
		if (current_frame > buffer_count-1){
			current_frame = buffer_count-1;
			return true;
		}
	}
	
	return false;
}

void vcontroller::move2previous2x()
{
	current_frame ? (--current_frame ? --current_frame : current_frame = 0) : current_frame = 0;
}

void vcontroller::move2previous1x()
{
	current_frame ? --current_frame : current_frame = 0;
}

void vcontroller::on_play()
{
	is_play = true;
}

bool vcontroller::Play()
{
	return is_play;
}

void vcontroller::off_play()
{
	is_play = false;
}

void vcontroller::move2forward1x()
{
	current_frame == buffer_count-1 ? current_frame = current_frame : ++current_frame;
}

void vcontroller::move2forward2x()
{
	current_frame == buffer_count-1 ? current_frame = current_frame : (++current_frame == buffer_count-1 ? current_frame = current_frame : ++current_frame);
}