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

// using std::chrono::time_point;
// using std::chrono::duration_cast;
// using std::chrono::microseconds;
// using std::chrono::seconds;
// using chrono = std::chrono::high_resolution_clock;

enum class MachineType
{
  Dummy,
  BruteForce,
  MonteCarlo,
  MCTS,
  MCTS_TT,
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
    } while (!bits.compare_exchange_weak(old_bits, new_bits, std::memory_order_relaxed,
                                         std::memory_order_relaxed));
  }
};

// A position on the board with it associated quality
struct Position
{
  Position(unsigned int board_row = 0, unsigned int board_column = 0,
           double avg_shortest_path_quality = 0.0, double opponent_avg_shortest_path_quality = 0.0,
           double quality = 0.0)
      : board_row{board_row},
        board_column{board_column},
        avg_shortest_path_quality{avg_shortest_path_quality},
        opponent_avg_shortest_path_quality{opponent_avg_shortest_path_quality},
        quality{quality}
  {
    double normalizer{avg_shortest_path_quality + opponent_avg_shortest_path_quality};
    if (normalizer != 0) winning_proba = avg_shortest_path_quality / normalizer;
  }

  // Row number on the game board
  unsigned int board_row;
  // Column number on the game board
  unsigned int board_column;
  // Id of the cell on the board board_row*board_size + board_column;
  unsigned int cell_id;
  // Node id in the graph (board_row+1)*(board_size+2) + (board_column+1);
  unsigned int node_id;
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
    lhs_proba /= (lhs.avg_shortest_path_quality + lhs.opponent_avg_shortest_path_quality);

    rhs_proba = rhs.avg_shortest_path_quality;
    rhs_proba /= (rhs.avg_shortest_path_quality + rhs.opponent_avg_shortest_path_quality);

    return lhs_proba < rhs_proba;
  }
};

struct HexMachineEngine
{
  HexMachineEngine(unsigned int board_size, MachineType mt,
                   unsigned int threads = 1 /*number of thread to spawn*/,
                   HexStatistics* stats_ptr = nullptr)
      : threads{threads},
        machine_type{mt},
        board_size{board_size},
        gen(rd()),
        uniform_distribution(std::uniform_int_distribution<int>(0, board_size - 1)),
        stats{stats_ptr}
  {
  }

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

  virtual void get_position(HexBoard& board, unsigned int& row, unsigned int& column,
                            unsigned int machine_player_id) = 0;

  // Modify the number of threads to be spawned
  void set_threads(unsigned int threads)
  {
    if (threads < 1) throw std::runtime_error{"The number of threads has to be at least 1."};
    this->threads = threads;
  }

  // Helper function for printing a duration
  void print_duration(std::chrono::time_point<std::chrono::high_resolution_clock> start,
                      std::chrono::time_point<std::chrono::high_resolution_clock> stop)
  {
    auto duration{std::chrono::duration_cast<std::chrono::microseconds>(stop - start)};
    auto sec{std::chrono::duration_cast<std::chrono::seconds>(duration)};
    auto us{duration - std::chrono::duration_cast<std::chrono::microseconds>(sec)};
    std::cout << "Duration : " << sec.count() << " s " << us.count() << " us" << std::endl;
  }

protected:
  HexStatistics* stats;

  friend struct Position;

  const unsigned int board_size;

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
  unsigned int threads{1};

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
  HexMachineDummy(unsigned int board_size) : HexMachineEngine(board_size, MachineType::Dummy, 1)
  {
  }

  void get_position(HexBoard& board, unsigned int& row, unsigned int& column,
                    unsigned int machine_player_id) override
  {
    // Stupid machine:
    // generate random numbers for the row and the column
    row = uniform_distribution(gen);
    column = uniform_distribution(gen);
  }
};

// Brute Force
// This is only for small boards (e.g. on a 8 cores computer selecting the first
// position of a 5x5 board would take about 12 minutes, and about 5 minutes for
// the second position)
struct HexMachineBF : HexMachineEngine
{
  HexMachineBF(unsigned int board_size, unsigned int threads = 1)
      : HexMachineEngine(board_size, MachineType::BruteForce, threads)
  {
  }

  // Brute-force: Test all available position
  void get_position(HexBoard& board, unsigned int& board_row, unsigned int& board_column,
                    unsigned int machine_player_id) override
  {
    std::vector<std::array<unsigned int, 2>> available_positions{};

    bf_task_call_count = 0;

    // Start chrono
    auto start{std::chrono::high_resolution_clock::now()};

    // Make a copy of the current board (should do it on the heap...)
    HexBoard board_copy{board};

    // Get all the available cells on the board
    board_copy.get_all_available_position(available_positions);

    // Prepare the storage for the stats:
    // store the available positions in the vector Positions_quality and the threads
    // for the bf_task will compute the quality of each position.
    results.Positions_quality.clear();
    for (auto pos : available_positions)
    {
      unsigned int board_row = pos[0];
      unsigned int board_column = pos[1];
      // std::cout << "(" << board_row << "," << board_column << "), ";
      Position position{board_row, board_column};
      position.cell_id = board_row * board_size + board_column;
      position.node_id = (board_row + 1) * (board_size + 2) + (board_column + 1);
      results.Positions_quality.push_back(position);
    }
    // std::cout << "\b\b \n";

    // Split the positions among threads for brute-force
    spawn_threads(available_positions, machine_player_id, board);

    Position best_position;
    double best_winning_quality{std::numeric_limits<double>::lowest()};
    for (auto position : results.Positions_quality)
    {
      if (position.quality > best_winning_quality)
      {
        best_winning_quality = position.quality;
        best_position = position;
      }
    }

    // for(auto position: results.Positions_quality) {
    //   std::cout << "Position (" << position.board_row << ","
    // 		<< position.board_column << ") quality: "
    // 		<< position.quality << std::endl;
    // }

    if (best_winning_quality < 0.0)
      std::cout << PlayersColors::color(machine_player_id) << " machine player"
                << " will definitively loose whatever it does!" << std::endl;

    board_row = best_position.board_row;
    board_column = best_position.board_column;

    // Stop the chrono
    auto stop{std::chrono::high_resolution_clock::now()};
    // Print the time taken in number of seconds + microseconds
    print_duration(start, stop);

    return;
  }

private:
  std::atomic<unsigned int> bf_task_call_count;

  struct Stats : HexMachineEngine::Stats
  {
    std::vector<Position> Positions_quality;

  } results;

  /*
   *  The 'task' to perform by a thread for the brute-force algorithm.
   *  This function is called by the n_choose_k() function when a combination
   *  of available positions for one player is formed.
   *  - Make a copy of the board
   *  - Place the tokens for the player with id 'player_id' on the board.
   *    The positions are given by one of the combination that is stored in
   *    'selected_positions' (see n_choose_k()).
   *  - Fill the remaining positions on the board with the opponent's color.
   *  - Compute the quality of each position in the "selected_positions":
   *       The quality of a position in the current combination is the sum
   *       of the shortest paths cost (all shortest path are considered) minus
   *       the quality of that position for the opponent.
   *       That is, since there can be only 1 winner, for a given filled board
   *       if the current player wins then its opponent looses and therefore the
   *       quality for the opponent is 0. On the other hand, if the current
   *       player looses, then its opponent wins and we compute the quality of
   *       the selected position for the opponent (i.e. sum of shortest path(s)
   *       cost for the opponent). Then the quality of the position the player
   *       is "0 - quality_for_the_opponent".
   *       So, at the end, when all possible combinations would have been evaluted
   *       each available cell on the board will have a certain quality and the one
   *       with the highest score will be selected.
   */
  void bf_task(const std::vector<std::array<unsigned int, 2>>& selected_positions, HexBoard& board,
               unsigned int player_id)
  {
    // Use a copy of the board (on the heap)
    std::unique_ptr<HexBoard> board_copy{new HexBoard(board)};

    // Place the selected position on the board and check if win
    board_copy->fill_with_color(player_id, selected_positions);

    // Fill all the remaining positions with the opponent color
    unsigned int opponent_id{(player_id == blue_player) ? red_player : blue_player};

    // Complete the board with opponent color
    board_copy->complete_with_color(opponent_id);

    // Check if it is a winning state
    double quality{0.0};
    Path shortest_path;
    bool winning_board{board_copy->has_won(player_id, &shortest_path, &quality)};

    // If it is not a winning state then it is a winning state for the opponent.
    // Then get the quality of that state for the opponent.
    double opponent_quality{0.0};
    bool opponent_winning_board{false};
    unsigned int missing_node_id;
    bool game_over{true};

    if (!winning_board)
    {
      opponent_winning_board = board_copy->has_won(opponent_id, nullptr, &opponent_quality);
      game_over = false;
    }
    else
    {
      // For a winning board, if all the cells but one in the shortest path
      // already belongs to the player, it means that the player wins just by
      // selecting this missing cell.
      unsigned int missing_position{0};
      Node missing_node;
      unsigned int node_value;
      for (auto node : shortest_path.route)
      {
        node_value = static_cast<unsigned int>(board.get_node_value(node.id));
        if (node_value != player_id)
        {
          missing_node_id = node.id;
          if (++missing_position == 2)
          {
            game_over = false;
            break;  // No need to look further
          }
        }
      }
    }

    // There is always one and exactly one winner
    if ((!winning_board) && (!opponent_winning_board))
      throw std::runtime_error{"There must be a winner"};

    unsigned int missing_cell_id;
    if (game_over)
    {
      // Get the cell id on the board
      unsigned int row{static_cast<unsigned int>(std::floor(missing_node_id / (board_size + 2)))};
      unsigned int column{missing_node_id - row * (board_size + 2)};
      missing_cell_id = {(row - 1) * board_size + (column - 1)};
      // std::cout << "Game over with cell id " << missing_cell_id
      //	<< " (" << row -1 << ", " << column -1 << ")" << std::endl;
    }

    // Store the first position and its associated qualities
    for (size_t i{0}; i < results.Positions_quality.size(); ++i)
    {
      for (size_t j{0}; j < selected_positions.size(); ++j)
      {
        unsigned int board_row{selected_positions[j][0]};
        unsigned int board_column{selected_positions[j][1]};
        unsigned int cell_id{board_row * board_size + board_column};

        if (results.Positions_quality[i].cell_id == cell_id)
        {
          if ((game_over) && (missing_cell_id == cell_id))
          {
            // Make sure that the missing cell to complete a path is selected
            double quality_max{std::numeric_limits<double>::max()};
            results.lock();
            results.Positions_quality[i].quality = quality_max;
            results.unlock();
          }
          else
          {
            results.lock();
            // When lost (i.e. quality==0 and opponent_quality>0) then the quality
            // of the selected cell is decreased by the quality of the opponent
            results.Positions_quality[i].quality += quality - opponent_quality;
            results.unlock();
          }
          break;
        }
      }
    }

    // Atomic counter for checking the task call
    bf_task_call_count++;

    return;
  }

  /*
   * spawn_threads() distributes among the threads the computation of all the
   * possible combinations of selecting k=floor((n+1)/2) positions among the
   * n available ones. The split is done here and the generation of the
   * combinations for each thread is done in the recursive function n_choose_k().
   * For each combination that it computes, n_choose_k() calls the function
   * bf_task() that evaluates the quality of each selected positions on the
   * board for that particular combination.
   * At the end, each available positions will have a quality score (sum of all
   * quality score for each combination) and the best position to move on will
   * be selected.
   */
  void spawn_threads(std::vector<std::array<unsigned int, 2>>& available_positions,
                     unsigned int machine_player_id, HexBoard& board)
  {
    std::vector<unsigned int> indexes;

    for (size_t i{0}; i < available_positions.size(); ++i) indexes.push_back(i);

    // Number of available positions
    unsigned int n{static_cast<unsigned int>(available_positions.size())};

    // Try all available positions. That is, if there are n positions available
    // then we must choose floor((n+1)/2) positions.
    unsigned int k{static_cast<unsigned int>(floor((n + 1.0) / 2.0))};

    // For T threads, T - 1 threads select a particular position that
    // other threads cannot select (i.e. one thread own a particular position).
    // And the last thread runs n_choose_k() on available positions that are not
    // owned by the T - 1 other threads.
    // So, the first thread selects the first index in the list and call bf_task
    // for all combination of k-1 indexes in the remaining n-1 ones (i.e.
    // offset = 1).
    // the second thread selects the second index in the list and call bf_task
    // for all combination of k-1 indexes in the remaining n-2 ones (i.e.
    // offset = 2).
    // etc.
    // The last thread, call bf_task for all combination of k elements among the
    // n - (T-1) remaining ones  (i.e. offset = T-1).
    std::vector<std::thread> spawned_thread;
    std::vector<unsigned int> chosen;

    unsigned int offset;

    unsigned int total_threads{std::min(threads, k)};
    unsigned int t{0};
    for (; t < total_threads - 1; ++t)
    {
      chosen.clear();
      chosen.push_back(indexes[t]);
      offset = t + 1;

      // n choose k-1 is done in the first T - 1 threads
      spawned_thread.push_back(std::thread(&HexMachineBF::choose, this, offset, k - 1, indexes,
                                           chosen, &HexMachineBF::bf_task, available_positions,
                                           machine_player_id, board));
    }

    // Last thread to spawn: n-(k-1) choose k
    chosen.clear();
    offset = t;
    spawned_thread.push_back(std::thread(&HexMachineBF::choose, this, offset, k, indexes, chosen,
                                         &HexMachineBF::bf_task, available_positions,
                                         machine_player_id, board));

    // Join threads
    for (auto& thrd : spawned_thread) thrd.join();
  }

  /*
   * Function that is called by the thread for brute-force.
   * Given n available positions on the board, k = floor((n+1)/2) are
   * choosen for the current player (identify by its id 'player_id').
   * For each of the possible combination the function 'f' will be called
   * to evaluate each position in the combination.
   * At the end the cell among the available one that will have the best score
   * will be choosen.
   * Note: when we say 'player' we mean of course the 'machine player' (i.e.
   *       the machine plays against itself or again a human player.
   */
  void choose(size_t offset, size_t k, std::vector<unsigned int> all_indexes,
              std::vector<unsigned int> chosen,
              void (HexMachineBF::*f)(const std::vector<std::array<unsigned int, 2>>&, HexBoard&,
                                      unsigned int),
              std::vector<std::array<unsigned int, 2>> positions, unsigned int player_id,
              HexBoard board)
  {
    n_choose_k(offset, k, all_indexes, chosen, f, positions, player_id, board);
  }

  /*
   * The work done in the thread for brute-force.
   * n_choose_k() is a recursive function that compute all combinations
   * of k positions among all available ones starting at a given offset.
   * e.g. if all_indexes = [11, 17, 33, 34, 44] then n_choose_k, with k = 3
   * and offset = 1, will call the function 'f' for all combination
   * of 3 position's indexes from [17, 33, 34, 44].
   */
  void n_choose_k(size_t offset, size_t k, std::vector<unsigned int>& all_indexes,
                  std::vector<unsigned int>& chosen,
                  void (HexMachineBF::*f)(const std::vector<std::array<unsigned int, 2>>&,
                                          HexBoard&, unsigned int),
                  std::vector<std::array<unsigned int, 2>>& positions, unsigned int player_id,
                  HexBoard& board)
  {
    if (k == 0)
    {
      // A combination is generated. Call f() to evaluate it.
      std::vector<std::array<unsigned int, 2>> selected_positions;
      selected_positions.clear();

      for (auto idx : chosen) selected_positions.push_back(positions[idx]);

      (this->*f)(selected_positions, board, player_id);

      return;
    }

    // Recursively generate a combination
    for (size_t i{offset}; i <= all_indexes.size() - k; ++i)
    {
      chosen.push_back(all_indexes[i]);
      n_choose_k(i + 1, k - 1, all_indexes, chosen, f, positions, player_id, board);
      chosen.pop_back();
    }
  }
};

#endif  // def HEX_MACHINE_ENGINE_H
