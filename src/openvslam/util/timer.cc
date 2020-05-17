
#include <chrono>  // NOLINT

#include "openvslam/util/timer.h"

namespace openvslam {
namespace util {


Timer::Timer() {
  start_ = std::chrono::high_resolution_clock::now();
}

void Timer::Reset() {
  start_ = std::chrono::high_resolution_clock::now();
}

double Timer::ElapsedTimeInSeconds() {
  const auto end = std::chrono::high_resolution_clock::now();
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
                 .count()) /
         (1000000.0);
}

}
}
