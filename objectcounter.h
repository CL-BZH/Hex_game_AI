#ifndef OBJECTCOUNTER_H
#define OBJECTCOUNTER_H

#include <mutex>

// Simple counter to track the number of object created
struct ObjectCounter {
  ObjectCounter(): count{0} {}

  ObjectCounter& operator++() {
    mu.lock();
    count++;
    mu.unlock();
    return *this;
  }
  ObjectCounter& operator--() {
    mu.lock();
    count--;
    mu.unlock();
    return *this;
  }
  // Cannot have a post-increment/decrement operator (because the copy
  // constructor is deleted since we have a mutex
  const ObjectCounter operator++(int dummy) = delete;
  const ObjectCounter operator--(int dummy) = delete;
  
  unsigned int operator()() const {
    return count;
  }
  
private:
  std::mutex mu;
  unsigned int count;
};

#endif //OBJECTCOUNTER_H
