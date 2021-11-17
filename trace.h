#ifndef TRACE_H
#define TRACE_H

#include <string>
#include <sstream>

// Just a small convinient class for tracing
struct Trace {
  Trace() {}

  template <class T>
  Trace& operator<<(T const& rhs) {
    sstr << rhs;
    return *this;
  }

  // Overload the () operator so that we can call trace()
  // instead of trace.print()
  void operator()() {
    print();
  }

  // Print what is currently in sstr and empty it.
  void print() {
    #ifdef _TRACE_MST
    // Print to console (add a newline)
    std::cout << sstr.str() << std::endl;
    // Empty it
    sstr.str(std::string());
    #endif
  }
  
private:
  std::stringstream sstr;
};

#endif //TRACE_H
