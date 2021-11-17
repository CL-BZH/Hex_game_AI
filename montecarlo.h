#ifndef MONTECARLO_H
#define MONTECARLO_H

#include <thread>
#include <mutex>
#include <chrono>
#include <thread>

#include <cstdio>

using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::seconds;
using chrono = std::chrono::high_resolution_clock;

#if defined( _DEBUG_MC )
#define MC_DEBUG_PRINT(x) printf x
#else
#define MC_DEBUG_PRINT(x)
#endif

// Generic class for running Monte-Carlo simulation
// This is an interface.
template <typename T>
struct MonteCarlo {

  MonteCarlo(T& algo, unsigned int threads=0): algo{algo}, threads{threads} {}

  // Modify the number of threads
  void set_threads(unsigned int threads) {
    if(threads < 1)
      throw std::runtime_error{"The number of threads has to be at least 1."};
    this->threads = threads;
  }

  // Read the current number of threads set for running the computation
  unsigned int get_threads() {
    return threads;
  }
  
  // Get the number of runs
  unsigned int get_runs() {
    return runs;
  }
  
  virtual void thread_work(unsigned int thread_number, unsigned int runs) = 0;

  virtual void compute_stats() = 0;
  virtual void show_stats() = 0;
  
  
  // Run the simulation
  void run(unsigned int runs=default_runs) {

    if(runs < 1)
      throw std::runtime_error{"The number of run should be at least 1!"};

    // At least 1 run per thread (we know that threads >= 1)
    if(runs < threads)
      threads = runs;

    // Disable cout
    std::cout.setstate(std::ios_base::failbit);

    // Start chrono
    auto start{chrono::now()};

    // Distribute the workload evenly between threads. e.g. To split 34 runs
    // between 4 threads, then the 2 first threads will have 9 runs and the
    // 2 last ones will have 8 runs
    spawn_threads(runs, threads);

    // Stop the chrono
    auto stop{chrono::now()};

    // Here there should remain only 1 instance
    MC_DEBUG_PRINT(("Number of algo object existing: %d\n", algo.instances()));
    
    // Re-enable cout
    std::cout.clear();

    // Print the time taken in number of seconds + microseconds
    print_duration(start, stop);
    
    // Compute the stats
    compute_stats();
  }

protected:
  
  // Algo to simulate
  T& algo;
  
  // Structure for the stats
  struct Stats{

    std::mutex stats_mutex;

    void lock() {
      stats_mutex.lock();
    }
    void unlock() {
      stats_mutex.unlock();
    }
    
    virtual void compute(double runs) = 0;
    virtual void show() const = 0;
  };
  
private:
  
  void spawn_threads(unsigned int runs, unsigned int threads) {
    
    // Minimum number of runs per threads
    unsigned int min_runs_per_thread{static_cast<unsigned int>
	(floor(static_cast<double>(runs)/threads))};
    
    // Compute remains = runs % threads (hence 'remains' is in [0, threads) )
    unsigned int remains{runs - min_runs_per_thread * threads};
    
    std::vector<std::thread> spawned_thread;
    
    for (unsigned int t{0}; t < threads; ++t) {
      unsigned int thread_runs{min_runs_per_thread};
      // Distribute the 'remains' runs between threads
      if(remains > 0) {
	--remains;
	++thread_runs;
      }
      MC_DEBUG_PRINT(("Spawn thread %d for %d runs\n", t, thread_runs));
      
      // Spawn threads to run the trials in parallel
      spawned_thread.push_back(std::thread(&MonteCarlo::thread_work,
					   this, t, thread_runs));
    }

    // Join threads
    for (auto& thrd: spawned_thread) {
      thrd.join();
    }
  }

  void print_duration(time_point<chrono> start, time_point<chrono> stop) {
    auto duration{duration_cast<microseconds>(stop - start)};
    auto sec{duration_cast<seconds>(duration)};
    auto us{duration - std::chrono::duration_cast<std::chrono::microseconds>(sec)};
    std::cout << "Duration : " << sec.count() << " s " << us.count()
	      << " us" << std::endl;
  }
  
  // Number of threads to spawn to run the simulation
  unsigned int threads;

  // Number of trials
  unsigned int runs{default_runs};
  
  // Default value for the number of runs
  static const unsigned int default_runs;

};

// Default number of run
template <typename T>
const unsigned int MonteCarlo<T>::default_runs{100};

#endif //MONTECARLO_H
