#ifndef HEX_MACHINE_ENGINE_H
#define HEX_MACHINE_ENGINE_H

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <math.h>

#include "graph.h"
#include "hex_board.h"

//using std::chrono::time_point;
//using std::chrono::duration_cast;
//using std::chrono::microseconds;
//using std::chrono::seconds;
//using chrono = std::chrono::high_resolution_clock;

enum class MachineType {
  Dummy,
  BruteForce,
  MonteCarlo,
  Undefined
};

// A position on the board with it associated quality
struct Position {
  Position(unsigned int board_row=0, unsigned int board_column=0,
	   double avg_shortest_path_quality=0.0,
	   double opponent_avg_shortest_path_quality=0.0, double quality=0.0):
    board_row{board_row}, board_column{board_column},
    avg_shortest_path_quality{avg_shortest_path_quality},
    opponent_avg_shortest_path_quality{opponent_avg_shortest_path_quality},
    quality{quality} {
      double normalizer{avg_shortest_path_quality +
			opponent_avg_shortest_path_quality};
      if(normalizer != 0)
	winning_proba = avg_shortest_path_quality / normalizer;
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
  
  Position& operator=(const Position& rhs) {
       board_row = rhs.board_row;
       board_column = rhs.board_column;
       avg_shortest_path_quality = rhs.avg_shortest_path_quality;
       opponent_avg_shortest_path_quality = rhs.opponent_avg_shortest_path_quality;
       return *this;
  }

  // Overload of the '<' operator for the priority queue.
  friend bool operator<(const Position& lhs, const Position& rhs) {
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


struct HexMachineEngine {

  HexMachineEngine(unsigned int board_size, MachineType mt,
		   unsigned int threads=1 /*number of thread to spawn*/):
    threads{threads}, machine_type{mt},
    board_size{board_size}, gen(rd()),
    uniform_distribution(std::uniform_int_distribution<int>(0, board_size-1)) {}
  
  virtual void get_position(HexBoard& board,
			    unsigned int& row, unsigned int& column,
			    unsigned int machine_player_id) = 0;

  
  // Modify the number of threads to be spawned
  void set_threads(unsigned int threads) {
    if(threads < 1)
      throw std::runtime_error{"The number of threads has to be at least 1."};
    this->threads = threads;
  }
  
  // Helper function for printing a duration
  void print_duration(std::chrono::time_point<std::chrono::high_resolution_clock> start,
		      std::chrono::time_point<std::chrono::high_resolution_clock> stop) {
    auto duration{std::chrono::duration_cast<std::chrono::microseconds>(stop - start)};
    auto sec{std::chrono::duration_cast<std::chrono::seconds>(duration)};
    auto us{duration - std::chrono::duration_cast<std::chrono::microseconds>(sec)};
    std::cout << "Duration : " << sec.count() << " s " << us.count()
	      << " us" << std::endl;
  }
  
protected:
  
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
  struct Stats {

    // Protect mutual access to shared data
    std::mutex stats_mutex;

    void lock() {
      stats_mutex.lock();
    }
    void unlock() {
      stats_mutex.unlock();
    }
  };

};

// Dummy machine.
// Just return random value for the row and the column in
// range [0, board_size - 1]
// It is the caller responsability to check the validity of the position
// (i.e. the position (row, column) is available on the board)
struct HexMachineDummy: HexMachineEngine {
  
  HexMachineDummy(unsigned int board_size):
    HexMachineEngine(board_size, MachineType::Dummy, 1) {}
  
  void get_position(HexBoard& board,
		    unsigned int& row, unsigned int& column,
		    unsigned int machine_player_id) override {
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
struct HexMachineBF: HexMachineEngine {

  HexMachineBF(unsigned int board_size,
	       unsigned int threads=1):
    HexMachineEngine(board_size, MachineType::BruteForce, threads) {}

  
   // Brute-force: Test all available position
  void get_position(HexBoard& board,
		    unsigned int& board_row, unsigned int& board_column,
		    unsigned int machine_player_id) override {
    
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
    for(auto pos: available_positions) {
      unsigned int board_row = pos[0];
      unsigned int board_column = pos[1];
      //std::cout << "(" << board_row << "," << board_column << "), ";
      Position position{board_row, board_column};
      position.cell_id = board_row*board_size + board_column;
      position.node_id = (board_row+1)*(board_size+2) + (board_column+1); 
      results.Positions_quality.push_back(position);
    }
    //std::cout << "\b\b \n";
   
    // Split the positions among threads for brute-force
    spawn_threads(available_positions, machine_player_id, board);

    Position best_position;
    double best_winning_quality{std::numeric_limits<double>::lowest()};
    for(auto position: results.Positions_quality) {
      if(position.quality > best_winning_quality) {
	best_winning_quality = position.quality;
	best_position = position;
      }
    }

    // for(auto position: results.Positions_quality) {
    //   std::cout << "Position (" << position.board_row << ","
    // 		<< position.board_column << ") quality: "
    // 		<< position.quality << std::endl;
    // }
    
    if(best_winning_quality < 0.0)
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
  
  struct Stats: HexMachineEngine::Stats {
    
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
  void bf_task(const std::vector<std::array<unsigned int, 2>>& selected_positions,
	       HexBoard& board, unsigned int player_id) {

    // Use a copy of the board (on the heap)
    std::unique_ptr<HexBoard> board_copy{new HexBoard(board)};

    //Place the selected position on the board and check if win
    board_copy->fill_with_color(player_id, selected_positions);

    // Fill all the remaining positions with the opponent color
    unsigned int opponent_id{(player_id == blue_player)?
	red_player : blue_player};
    
    //Complete the board with opponent color
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
    
    if(!winning_board) {
      opponent_winning_board = board_copy->has_won(opponent_id, nullptr,
						   &opponent_quality);
      game_over = false;
    } else {
      // For a winning board, if all the cells but one in the shortest path
      // already belongs to the player, it means that the player wins just by
      // selecting this missing cell.
      unsigned int missing_position{0};
      Node missing_node;
      unsigned int node_value;
      for(auto node: shortest_path.route) {
	node_value = static_cast<unsigned int>(board.get_node_value(node.id));
	if(node_value != player_id) {
	  missing_node_id = node.id;
	  if(++missing_position == 2) {
	    game_over = false;
	    break; //No need to look further
	  }
	}
      }
    }

    // There is always one and exactly one winner
    if((!winning_board) && (!opponent_winning_board))
      throw std::runtime_error{"There must be a winner"};

    unsigned int missing_cell_id;
    if(game_over) {
      // Get the cell id on the board
      unsigned int row{static_cast<unsigned int>(std::floor(missing_node_id /
							    (board_size+2)))};
      unsigned int column{missing_node_id - row*(board_size+2)};
      missing_cell_id = {(row - 1) * board_size + (column - 1)};
      //std::cout << "Game over with cell id " << missing_cell_id
      //	<< " (" << row -1 << ", " << column -1 << ")" << std::endl;
      
    }
    
    // Store the first position and its associated qualities
    for(size_t i{0}; i < results.Positions_quality.size(); ++i) {
      for(size_t j{0}; j < selected_positions.size(); ++j) {
	unsigned int board_row{selected_positions[j][0]};
	unsigned int board_column{selected_positions[j][1]};
	unsigned int cell_id{board_row * board_size + board_column};
    
	if(results.Positions_quality[i].cell_id == cell_id) {
	  if((game_over) && (missing_cell_id == cell_id)) {
	    // Make sure that the missing cell to complete a path is selected
	    double quality_max{std::numeric_limits<double>::max()};
	    results.lock();
	    results.Positions_quality[i].quality = quality_max;
	    results.unlock();
	  } else {
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
  void spawn_threads(std::vector<std::array<unsigned int, 2>>&
		     available_positions, unsigned int machine_player_id,
		     HexBoard& board) {

    std::vector<unsigned int> indexes;
    
    for(size_t i{0}; i < available_positions.size(); ++i)
      indexes.push_back(i);

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
    for (;t < total_threads - 1; ++t) {
      chosen.clear();
      chosen.push_back(indexes[t]);
      offset = t+1;
      
      // n choose k-1 is done in the first T - 1 threads 
      spawned_thread.push_back(std::thread(&HexMachineBF::choose, this,
					   offset, k - 1,
					   indexes, chosen,
					   &HexMachineBF::bf_task,
					   available_positions,
					   machine_player_id, board));
    }
    
    // Last thread to spawn: n-(k-1) choose k
    chosen.clear();
    offset = t;
    spawned_thread.push_back(std::thread(&HexMachineBF::choose, this,
					 offset, k,
					 indexes, chosen,
					 &HexMachineBF::bf_task,
					 available_positions,
					 machine_player_id, board));
    
    // Join threads
    for (auto& thrd: spawned_thread)
      thrd.join();

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
  void choose(size_t offset, size_t k,
	      std::vector<unsigned int> all_indexes,
	      std::vector<unsigned int> chosen,
	      void (HexMachineBF::*f)
	      (const std::vector<std::array<unsigned int, 2>>&,
	       HexBoard&, unsigned int),
	      std::vector<std::array<unsigned int, 2>> positions,
	      unsigned int player_id, HexBoard board) {

    n_choose_k(offset, k, all_indexes, chosen, f,
	       positions, player_id, board);
  }

  /*
   * The work done in the thread for brute-force.
   * n_choose_k() is a recursive function that compute all combinations
   * of k positions among all available ones starting at a given offset.
   * e.g. if all_indexes = [11, 17, 33, 34, 44] then n_choose_k, with k = 3
   * and offset = 1, will call the function 'f' for all combination
   * of 3 position's indexes from [17, 33, 34, 44].
   */
  void n_choose_k(size_t offset, size_t k,
		  std::vector<unsigned int>& all_indexes,
		  std::vector<unsigned int>& chosen,
		  void (HexMachineBF::*f)
		  (const std::vector<std::array<unsigned int, 2>>&,
		   HexBoard&, unsigned int),
		  std::vector<std::array<unsigned int, 2>>& positions,
		  unsigned int player_id, HexBoard& board) {
    
    if (k == 0) {
      // A combination is generated. Call f() to evaluate it.
      std::vector<std::array<unsigned int, 2>> selected_positions;
      selected_positions.clear();
      
      for(auto idx: chosen)
	selected_positions.push_back(positions[idx]);
      
      (this->*f)(selected_positions, board, player_id);
      
      return;
    }

    // Recursively generate a combination
    for (size_t i{offset}; i <= all_indexes.size() - k; ++i) {
      chosen.push_back(all_indexes[i]);
      n_choose_k(i+1, k-1, all_indexes, chosen, f, positions, player_id,
		 board);
      chosen.pop_back();
    }
  }

};

// Monte-Carlo IA
struct HexMachineMcIA: HexMachineEngine {
  
  HexMachineMcIA(unsigned int board_size,
		unsigned int threads=1 /*number of thread to spawn*/):
    #ifdef _BF_SUP
    brute_force_machine{board_size, threads},
    #endif
    HexMachineEngine(board_size, MachineType::MonteCarlo, threads) {}

  void get_position(HexBoard& board,
		    unsigned int& board_row, unsigned int& board_column,
		    unsigned int machine_player_id) override {
#ifdef _BF_SUP
    // If the number of available cells on the board is less than
    // the limit sup then use brute-force
    if(board.nb_available_cells() < _BF_SUP) {
      //std::cout << "[B.F] Number of available cell for "
      //		<< PlayersColors::color(machine_player_id) << ": "
      //		<< board.nb_available_cells() << std::endl;
      
      brute_force_machine.get_position(board, board_row, board_column,
				       machine_player_id);
      return;
    }
#endif
    // else use Monte-Carlo method
    get_position_mc(board, board_row, board_column, machine_player_id);
     
   }
  
  /*
   *  Get the best candidate based on a Monte-Carlo method.
   *  Run T times the Monte-Carlo trial for each available position on
   *  the board. Where trial i for that position consist of:
   *  - Selecting the position
   *  - Filling the remaining positions on the board by randomly distributing
   *    the blue and red tokens
   *  - Computing the quality of the position:
   *       The quality of the position for the current filled board is the sum
   *       of the shortest paths cost (all shortest path are considered)
   *  - Computing the quality of the position for the opponent.
   *  - Updating the priority queue for selecting the position which gives the
   *    best (estimated) probability of winning. (see the overload of '<' in
   *    the class 'Position')
   */
  void get_position_mc(HexBoard& board,
		       unsigned int& board_row, unsigned int& board_column,
		       unsigned int machine_player_id) {

    unsigned int runs{get_number_runs(board)};
	
    // Store all possible positions with their probability of winning
    std::priority_queue<Position> best_positions;
    
    bool game_end{false};
    bool got_position{false};
    
    unsigned int current_board_row{0};

    // Start chrono
    auto start{std::chrono::high_resolution_clock::now()};
    
    while(current_board_row < board_size) {
      
      unsigned int current_board_column{0};
      
      while(current_board_column < board_size) {
	  
	// * Selecting the position
	got_position = board.get_first_available_position(current_board_row,
							  current_board_column,
							  machine_player_id,
							  game_end);

	// We went through all the board
	if(!got_position) {
	  // Normal case: we reach the end of the board and the cell
	  // is already allocated. Then we will have current_board_column
	  // back to 0 and current_board_row equal board_size. Using 'break'
	  // we will exit the 2 'while' loops.
	  if((current_board_row == board_size) &&
	     (current_board_column == 0))
	    break;
	  // else: there is a bug!
	  std::stringstream err;
	  err << PlayersColors::color(machine_player_id)
	      << " didn't get a position. Starting from row "
	      << current_board_row << ", column "
	      << current_board_column << std::endl;
	  throw std::runtime_error{err.str()};
	}

	// The currently selected cell on the board
	unsigned int selected_node_row{current_board_row+1};
	unsigned int selected_node_column{current_board_column+1};
	unsigned int selected_node_id = selected_node_row*(board_size+2)
	  + selected_node_column;
	
	// If win with the current selection then use it
	if(game_end) {
	  board_row = current_board_row;
	  board_column = current_board_column;
	  // Release the selected position for it to be selected by
	  // select() in machine_play()
	  board.release_board_node(selected_node_id);
	  return;
	}

	// Now board_row, board_column represent the first cell that was available
	// when going throught the boards' cells from top to bottom - left to 
	// right. The cell is selected by the machine player as a candidate. Then,
	// each thread will evaluate the quality of that selection
	
	// * Filling the remaining positions on the board by randomly
	//   distributing the blue and red cells. i.e. Run Monte-Carlo simulation
	//   (see mc_task()) for this position in multiple threads.
	spawn_threads(runs, threads, selected_node_id, machine_player_id, board);
	
	// Release the selected candidate position
	board.release_board_node(selected_node_id);

	// Update the average quality for the position from the quality that
	// each thread computed (see mc_task())
	double player_avg_quality{
	  std::accumulate(results.player_quality_sum.begin(),
			  results.player_quality_sum.end(), 0.0) / runs};
      
	double opponent_avg_quality{
	  std::accumulate(results.opponent_quality_sum.begin(),
			  results.opponent_quality_sum.end(), 0.0) / runs};

	// Reset for the next candidate position quality computation
	results.player_quality_sum.clear();
	results.opponent_quality_sum.clear();

	// Store the position with its quality in the queue
	Position pos{current_board_row, current_board_column,
		     player_avg_quality, opponent_avg_quality};
	
	best_positions.push(pos);

	//  Select next position (i.e. go to next column)
	++current_board_column;
      }
      // Select next position (i.e. go to next row, first column)
      ++current_board_row;
    }

    // Stop the chrono
    auto stop{std::chrono::high_resolution_clock::now()};
    
    // Print the time taken in number of seconds + microseconds
    print_duration(start, stop);

    // Select the best candidate position (i.e. the cell's selection that
    // gives the best estimated probability of winning) for the player.
    Position best_pos{best_positions.top()};
    board_row = best_pos.board_row;
    board_column = best_pos.board_column;

#ifndef _NCURSES
    const double winning_proba{best_pos.winning_proba};
    int percent{(static_cast<int>(winning_proba * 10000.0))/100};
    std::cout << PlayersColors::color(machine_player_id)
	      << " machine: I estimate that I have " << percent
	      << "% chance of winning ";
    if(percent < 30)
      std::cout << ":((";
    else if(percent < 40)
      std::cout << ":(";
    else if(percent < 50)
      std::cout << ":|";
    std::cout  << std::endl;
#endif
    
  }
  
  // Set the number of runs for the Monte-Carlo simulation in function
  // of the board's size and the number of available cells.
  unsigned int get_number_runs(const HexBoard& board) {
    // Number of cells on a board
    const unsigned int total_cells{board_size*board_size};
    // max number of runs: make it a factor of the board's size.
    // The bigger the board the more states there is, so more MC runs are needed.
    const unsigned int max_runs{total_cells * max_runs_factor};

    // Shift the center of the bell curve: the maximum number of runs is not used
    // at the very beginning of the game otherwise it would take too long.
    // When there is a bit less available cells (i.e. less possible states) then
    // the maximum number of run is used. 
    const double center{static_cast<double>(7 * total_cells / 8)};
    double available_cells{static_cast<double>(board.nb_available_cells())};

    const double sigma_square{1.0 * total_cells};
    
    unsigned int runs = static_cast<unsigned int>(bell_shape(board, max_runs,
							     center, sigma_square,
							     available_cells));

    //std::cout << "runs: " << runs << std::endl;
    
    runs = (runs < 100)? 100: runs;
    return runs;
  }

 private:

  struct Stats: HexMachineEngine::Stats {
    
    std::vector<unsigned int> player_quality_sum;
    std::vector<unsigned int> opponent_quality_sum;
    
  } results;

  // Factor to determine the maximum number of runs per MC trial
  static constexpr unsigned int max_runs_factor{25};
  
  #ifdef _BF_SUP
  HexMachineBF brute_force_machine;
  #endif
  
  // The 'task' performed by a thread for the Monte-Carlo simulation
  // Each thread receives the cuurent board with a particular position
  // selected as a candidate for a move for the player identified by 'player_id'.
  // Then it performs the following task:
  //    1. Make a copy of the board
  //    2. Do 'runs' times:
  //       a. Randonly fill the board with tokens for the opponent
  //          and the current player
  //       b. Compute the quality of the candidate cell for that filled board
  //          and store it.
  //       c. Clean the cells filled in a.
  void mc_task(unsigned int thread_number, unsigned int runs,
		   unsigned int selected_node_id, unsigned int player_id,
		   const HexBoard& board) {

    // Each thread has its own generator...
    std::mt19937 gen(rd());

    //std::cout << "Thread " << thread_number << " starts " << runs << " runs\n";
    
    // Make a copy (on the heap) of the current board.
    std::unique_ptr<HexBoard> board_copy{new HexBoard(board)};
    
    // Switch players:
    // Since the player identified by 'player_id' just select a candidate cell
    // for its move it is now the turn of the opponent to select a position.
    // So, rand_fill_board() will start with the opponent and alternate between
    // the two players
    unsigned int opponent_id = (player_id == blue_player)?
      red_player : blue_player;
    
    std::vector<double> player_quality;
    std::vector<double> opponent_quality;
    
    while(runs) {
      // Fill the board by randomly selecting red and blue positions among
      // the available ones
      std::vector<unsigned int> available_nodes_ids{};
      board_copy->rand_fill_board(opponent_id, gen, available_nodes_ids);
      
      // Getting the quality of that board for the current player
      // (see function has_won() in hex_board.h for the definition of 'quality') 
      double quality{0.0};
      bool winning_board{board_copy->has_won(player_id, nullptr, &quality)};
      player_quality.push_back(quality);

      // Getting the quality of that board for the opponent
      // (recall: there can be only 1 winner)
      quality = 0.0;
      if(!winning_board)
	board_copy->has_won(opponent_id, nullptr, &quality);
      opponent_quality.push_back(quality);
      
      // Release the nodes for reusing the board
      board_copy->release_board_nodes(available_nodes_ids);
      
      --runs;
    }
    
    // 
    double player_quality_total{std::accumulate(player_quality.begin(),
					    player_quality.end(), 0.0)};

    // Total number of shortest path for the opponent if select the
    // current position
    double opponent_quality_total{std::accumulate(opponent_quality.begin(),
						  opponent_quality.end(),
						  0.0)};

    // Store the quality of a selection computed by the thread
    results.lock();
    results.player_quality_sum.push_back(player_quality_total);
    results.opponent_quality_sum.push_back(opponent_quality_total);
    results.unlock();
  }

  // Spawn threads for the Monte-Carlo simulation
  void spawn_threads(unsigned int runs, unsigned int threads,
		     unsigned int selected_node_id,
		     unsigned int machine_player_id,
		     HexBoard& board) {
    
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
      
      // Spawn threads to run the trials in parallel
      spawned_thread.push_back(std::thread(&HexMachineMcIA::mc_task,
					   this, t, thread_runs,
					   selected_node_id,
					   machine_player_id,
					   board));
    }

    // Join threads
    for (auto& thrd: spawned_thread) {
      thrd.join();
    }
  }

  // Generate a bell-shape curve
  unsigned int bell_shape(const HexBoard& board, double max,
			  double center, double sigma_sqr, double x) const {

    const double delta{center - x};
    const double delta_sqr{delta*delta};
    
    const double bell_shape{exp(-0.5 * delta_sqr / sigma_sqr)};
    const double y{max * bell_shape};
    
    return y;
  }
  
};


#endif //def HEX_MACHINE_ENGINE_H
