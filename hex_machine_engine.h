#ifndef HEX_MACHINE_ENGINE_H
#define HEX_MACHINE_ENGINE_H

#include <math.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "graph.h"
#include "hex_board.h"
#include "hex_statistics.h"

enum class MachineType
{
  Dummy,
  BruteForce,
  MonteCarlo,
  MCTS,
  Undefined
};

class AtomicDouble
{
private:
  std::atomic<uint64_t> bits;

  static uint64_t double_to_bits(double d)
  {
    uint64_t b;
    std::memcpy(&b, &d, sizeof(double));
    return b;
  }

  static double bits_to_double(uint64_t b)
  {
    double d;
    std::memcpy(&d, &b, sizeof(double));
    return d;
  }

public:
  AtomicDouble() : bits(0)
  {
  }
  explicit AtomicDouble(double d) : bits(double_to_bits(d))
  {
  }

  double load() const
  {
    return bits_to_double(bits.load(std::memory_order_relaxed));
  }

  void store(double d)
  {
    bits.store(double_to_bits(d), std::memory_order_relaxed);
  }

  void add(double d)
  {
    uint64_t old_bits = bits.load(std::memory_order_relaxed);
    uint64_t new_bits;
    do
    {
      double old_val = bits_to_double(old_bits);
      double new_val = old_val + d;
      new_bits = double_to_bits(new_val);
    } while (!bits.compare_exchange_weak(old_bits, new_bits,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed));
  }
};

// A position on the board with it associated quality
struct Position
{
  Position(uint board_row = 0, uint board_column = 0,
           double avg_shortest_path_quality = 0.0,
           double opponent_avg_shortest_path_quality = 0.0,
           double quality = 0.0)
      : board_row{board_row},
        board_column{board_column},
        avg_shortest_path_quality{avg_shortest_path_quality},
        opponent_avg_shortest_path_quality{opponent_avg_shortest_path_quality},
        quality{quality}
  {
    double normalizer{avg_shortest_path_quality +
                      opponent_avg_shortest_path_quality};
    if (normalizer != 0) winning_proba = avg_shortest_path_quality / normalizer;
  }

  // Row number on the game board
  uint board_row;
  // Column number on the game board
  uint board_column;
  // Id of the cell on the board board_row*board_size + board_column;
  uint cell_id;
  // Node id in the graph (board_row+1)*(board_size+2) + (board_column+1);
  uint node_id;
  // Position's quality (indicate the "chance" a selected position has to
  // lead to a winning path)
  double quality;
  // Estimation of the probability of winning
  double winning_proba{0};

  double avg_shortest_path_quality;
  double opponent_avg_shortest_path_quality;

  Position& operator=(const Position& rhs)
  {
    board_row = rhs.board_row;
    board_column = rhs.board_column;
    avg_shortest_path_quality = rhs.avg_shortest_path_quality;
    opponent_avg_shortest_path_quality = rhs.opponent_avg_shortest_path_quality;
    return *this;
  }

  // Overload of the '<' operator for the priority queue.
  friend bool operator<(const Position& lhs, const Position& rhs)
  {
    // Get the (estimated) winning probability for the lhs position
    // and the rhs position.
    double lhs_proba{1.0};
    double rhs_proba{1.0};

    lhs_proba = lhs.avg_shortest_path_quality;
    lhs_proba /= (lhs.avg_shortest_path_quality +
                  lhs.opponent_avg_shortest_path_quality);

    rhs_proba = rhs.avg_shortest_path_quality;
    rhs_proba /= (rhs.avg_shortest_path_quality +
                  rhs.opponent_avg_shortest_path_quality);

    return lhs_proba < rhs_proba;
  }
};

struct HexMachineEngine
{
  HexMachineEngine(uint board_size, MachineType mt,
                   uint threads = 1 /*number of thread to spawn*/,
                   HexStatistics* stats_ptr = nullptr)
      : threads{threads},
        machine_type{mt},
        board_size{board_size},
        gen(rd()),
        uniform_distribution(
            std::uniform_int_distribution<int>(0, board_size - 1)),
        stats{stats_ptr}
  {
  }

  // Virtual destructor
  virtual ~HexMachineEngine() = default;

  virtual std::string get_name() const
  {
    switch (machine_type)
    {
      case MachineType::MonteCarlo:
        return "MonteCarlo";
      case MachineType::MCTS:
        return "MCTS";
      case MachineType::BruteForce:
        return "BruteForce";
      case MachineType::Dummy:
        return "Dummy";
      default:
        return "Unknown";
    }
  }

  virtual void get_position(HexBoard& board, uint& row, uint& column,
                            uint machine_player_id) = 0;

  // Modify the number of threads to be spawned
  void set_threads(uint threads)
  {
    if (threads < 1)
      throw std::runtime_error{"The number of threads has to be at least 1."};
    this->threads = threads;
  }

  // Helper function for printing a duration
  void print_duration(
      std::chrono::time_point<std::chrono::high_resolution_clock> start,
      std::chrono::time_point<std::chrono::high_resolution_clock> stop)
  {
    auto duration{
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)};
    auto sec{std::chrono::duration_cast<std::chrono::seconds>(duration)};
    auto us{duration -
            std::chrono::duration_cast<std::chrono::microseconds>(sec)};
    std::cout << "Duration : " << sec.count() << " s " << us.count() << " us"
              << std::endl;
  }

protected:
  HexStatistics* stats;

  friend struct Position;

  const uint board_size;

  // Store the type of machine
  MachineType machine_type;

  // obtain a random number from hardware
  std::random_device rd;

  // generator
  std::mt19937 gen;

  // Uniform distibution over the range [0, board_size - 1]
  // Note: std::uniform_int_distribution<> is inclusive
  std::uniform_int_distribution<int> uniform_distribution;

  // number of threads (Only 1 by default)
  uint threads{1};

  // Structure for the stats
  struct Stats
  {
    // Protect mutual access to shared data
    std::mutex stats_mutex;

    void lock()
    {
      stats_mutex.lock();
    }
    void unlock()
    {
      stats_mutex.unlock();
    }
  };
};

// Dummy machine.
// Just return random value for the row and the column in
// range [0, board_size - 1]
// It is the caller responsability to check the validity of the position
// (i.e. the position (row, column) is available on the board)
struct HexMachineDummy : HexMachineEngine
{
  HexMachineDummy(uint board_size)
      : HexMachineEngine(board_size, MachineType::Dummy, 1)
  {
  }

  void get_position(HexBoard& board, uint& row, uint& column,
                    uint machine_player_id) override
  {
    // Stupid machine:
    // generate random numbers for the row and the column
    row = uniform_distribution(gen);
    column = uniform_distribution(gen);
  }
};

#endif  // def HEX_MACHINE_ENGINE_H
