// created by Steffen Urban, urbste@gmail.com, May 2020
#ifndef OPENVSLAM_UTIL_TIMER_H
#define OPENVSLAM_UTIL_TIMER_H

#include <chrono>

namespace openvslam {
namespace util {

// Taken fron TheiaSfM
// Measures the elapsed wall clock time.
class Timer {
 public:
  // The constuctor starts a timer.
  Timer();

  // Resets the timer to zero.
  void Reset();

  // Measures the elapsed time since the last call to Reset() or the
  // construction of the object if Reset() has not been called. The resolution
  // of this timer is microseconds.
  double ElapsedTimeInSeconds();

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}
}
#endif
