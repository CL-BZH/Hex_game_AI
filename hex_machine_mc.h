#ifndef HEX_MACHINE_MC_H
#define HEX_MACHINE_MC_H

#include "hex_board.h"
#include "hex_machine_engine.h"

// Monte-Carlo IA
struct HexMachineMcIA : HexMachineEngine
{
  HexMachineMcIA(uint board_size,
                 uint threads = 1 /*number of thread to spawn*/,
                 bool quiet_mode = false, HexStatistics* stats_ptr = nullptr)
      :
#ifdef _BF_SUP
        brute_force_machine{board_size, threads},
#endif
        HexMachineEngine(board_size, MachineType::MonteCarlo, threads,
                         stats_ptr),
        quiet_mode(quiet_mode)
  {
  }

  void get_position(HexBoard& board, uint& board_row, uint& board_column,
                    uint machine_player_id) override
  {
#ifdef _BF_SUP
    // If the number of available cells on the board is less than
    // the limit sup then use brute-force
    if (board.get_nb_available_cells() < _BF_SUP)
    {
      brute_force_machine.get_position(board, board_row, board_column,
                                       machine_player_id);
      return;
    }
#endif
    // else use Monte-Carlo method
    get_position_mc(board, board_row, board_column, machine_player_id);
  }

  std::string get_name() const override
  {
    return "MonteCarlo";
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
  void get_position_mc(HexBoard& board, uint& board_row, uint& board_column,
                       uint machine_player_id)
  {
    uint runs{get_number_runs(board)};

    // Store all possible positions with their probability of winning
    std::priority_queue<Position> best_positions;

    bool game_end{false};
    bool got_position{false};

    uint current_board_row{0};

#ifndef _NCURSES
    if (!quiet_mode)
    {
      std::cout << PlayersColors::color(machine_player_id) << " MC: Running "
                << runs << " iterations in " << threads << " threads"
                << std::endl;
    }
#endif

    // Start chrono
    auto start{std::chrono::high_resolution_clock::now()};

    while (current_board_row < board_size)
    {
      uint current_board_column{0};

      while (current_board_column < board_size)
      {
        // Copy of the board to work on
        HexBoard board_copy(board);

        // * Selecting the position
        got_position = board_copy.get_first_available_position(
            current_board_row, current_board_column, machine_player_id,
            game_end);

        // We went through all the board
        if (!got_position)
        {
          // Normal case: we reach the end of the board and the cell
          // is already allocated. Then we will have current_board_column
          // back to 0 and current_board_row equal board_size. Using 'break'
          // we will exit the 2 'while' loops.
          if ((current_board_row == board_size) && (current_board_column == 0))
            break;
          // else: there is a bug!
          std::stringstream err;
          err << PlayersColors::color(machine_player_id)
              << " didn't get a position. Starting from row "
              << current_board_row << ", column " << current_board_column
              << std::endl;
          throw std::runtime_error{err.str()};
        }

        // The currently selected cell on the board
        uint selected_node_row{current_board_row + 1};
        uint selected_node_column{current_board_column + 1};
        uint selected_node_id =
            selected_node_row * (board_size + 2) + selected_node_column;

        // If win with the current selection then use it
        if (game_end)
        {
          board_row = current_board_row;
          board_column = current_board_column;
          // Release the selected position for it to be selected by
          // select() in machine_play()
          board_copy.release_board_node(selected_node_id);
          return;
        }

        // Now board_row, board_column represent the first cell that was
        // available when going throught the boards' cells from top to bottom -
        // left to right. The cell is selected by the machine player as a
        // candidate. Then, each thread will evaluate the quality of that
        // selection

        // * Filling the remaining positions on the board by randomly
        //   distributing the blue and red cells. i.e. Run Monte-Carlo
        //   simulation (see mc_task()) for this position in multiple threads.
        HexBoard board_tmp(board_copy);
        spawn_threads(runs, threads, selected_node_id, machine_player_id,
                      board_tmp);

        // Update the average quality for the position from the quality that
        // each thread computed (see mc_task())
        double player_avg_quality{
            std::accumulate(results.player_quality_sum.begin(),
                            results.player_quality_sum.end(), 0.0) /
            runs};

        double opponent_avg_quality{
            std::accumulate(results.opponent_quality_sum.begin(),
                            results.opponent_quality_sum.end(), 0.0) /
            runs};

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

    // Select the best candidate position (i.e. the cell's selection that
    // gives the best estimated probability of winning) for the player.
    Position best_pos{best_positions.top()};
    board_row = best_pos.board_row;
    board_column = best_pos.board_column;

    // Stop the chrono
    auto stop{std::chrono::high_resolution_clock::now()};

    if (!quiet_mode)
    {
      // Print the time taken in number of seconds + microseconds
      print_duration(start, stop);
    }
    else if (stats != nullptr)
    {
      auto duration{
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start)};
      std::string engine_name =
          get_name() + "_" +
          std::string(machine_player_id == blue_player ? "Blue" : "Red");
      stats->record_move_time(engine_name, duration);
    }

#ifndef _NCURSES
    if (!quiet_mode)
    {
      const double winning_proba{best_pos.winning_proba};
      int percent{(static_cast<int>(winning_proba * 10000.0)) / 100};
      std::cout << PlayersColors::color(machine_player_id)
                << " machine: I estimate that I have " << percent
                << "% chance of winning ";
      if (percent < 30)
        std::cout << ":((";
      else if (percent < 40)
        std::cout << ":(";
      else if (percent < 50)
        std::cout << ":|";
      std::cout << std::endl;
    }
#endif
  }

  // Set the number of runs for the Monte-Carlo simulation in function
  // of the board's size and the number of available cells.
  uint get_number_runs(const HexBoard& board)
  {
    // Number of cells on a board
    const uint total_cells{board_size * board_size};
    // max number of runs: make it a factor of the board's size.
    // The bigger the board the more states there is, so more MC runs are
    // needed.
    const uint max_runs{total_cells * max_runs_factor};

    // Shift the center of the bell curve: the maximum number of runs is not
    // used at the very beginning of the game otherwise it would take too long.
    // When there is a bit less available cells (i.e. less possible states) then
    // the maximum number of run is used.
    const double center{static_cast<double>(7 * total_cells / 8)};
    double available_cells{static_cast<double>(board.get_nb_available_cells())};

    const double sigma_square{1.0 * total_cells};

    uint runs = static_cast<uint>(
        bell_shape(board, max_runs, center, sigma_square, available_cells));
    runs = (runs < 100) ? 100 : runs;
    return runs;
  }

private:
  bool quiet_mode;

  struct Stats : HexMachineEngine::Stats
  {
    std::vector<uint> player_quality_sum;
    std::vector<uint> opponent_quality_sum;

  } results;

  // Factor to determine the maximum number of runs per MC trial
  static constexpr uint max_runs_factor{35};

#ifdef _BF_SUP
  HexMachineBF brute_force_machine;
#endif

  // The 'task' performed by a thread for the Monte-Carlo simulation
  // Each thread receives the current board with a particular position
  // selected as a candidate for a move for the player identified by
  // 'player_id'. Then it performs the following task:
  //    1. Make a copy of the board
  //    2. Do 'runs' times:
  //       a. Randonly fill the board with tokens for the opponent
  //          and the current player
  //       b. Compute the quality of the candidate cell for that filled board
  //          and store it.
  //       c. Clean the cells filled in a.
  void mc_task(uint thread_number, uint runs, uint selected_node_id,
               uint player_id, const HexBoard& board)
  {
    // Each thread has its own generator...
    std::mt19937 gen(rd());

    // Make a copy (on the heap) of the current board.
    std::unique_ptr<HexBoard> board_copy{new HexBoard(board)};

    // Switch players:
    // Since the player identified by 'player_id' just select a candidate cell
    // for its move it is now the turn of the opponent to select a position.
    // So, rand_fill_board() will start with the opponent and alternate between
    // the two players
    uint opponent_id = (player_id == blue_player) ? red_player : blue_player;

    std::vector<double> player_quality;
    std::vector<double> opponent_quality;

    while (runs)
    {
      // Fill the board by randomly selecting red and blue positions among
      // the available ones
      std::vector<uint> available_nodes_ids{};
      board_copy->rand_fill_board(opponent_id, gen, available_nodes_ids);

      // Getting the quality of that board for the current player
      // (see function has_won() in hex_board.h for the definition of 'quality')
      double quality{0.0};
      bool winning_board{board_copy->has_won(player_id, nullptr, &quality)};
      player_quality.push_back(quality);

      // Getting the quality of that board for the opponent
      // (recall: there can be only 1 winner)
      quality = 0.0;
      if (!winning_board) board_copy->has_won(opponent_id, nullptr, &quality);
      opponent_quality.push_back(quality);

      // Release the nodes for reusing the board
      board_copy->release_board_nodes(available_nodes_ids);

      --runs;
    }

    //
    double player_quality_total{
        std::accumulate(player_quality.begin(), player_quality.end(), 0.0)};

    // Total number of shortest path for the opponent if select the
    // current position
    double opponent_quality_total{
        std::accumulate(opponent_quality.begin(), opponent_quality.end(), 0.0)};

    // Store the quality of a selection computed by the thread
    results.lock();
    results.player_quality_sum.push_back(player_quality_total);
    results.opponent_quality_sum.push_back(opponent_quality_total);
    results.unlock();
  }

  // Spawn threads for the Monte-Carlo simulation
  void spawn_threads(uint runs, uint threads, uint selected_node_id,
                     uint machine_player_id, HexBoard& board)
  {
    // Minimum number of runs per threads
    uint min_runs_per_thread{
        static_cast<uint>(floor(static_cast<double>(runs) / threads))};

    // Compute remains = runs % threads (hence 'remains' is in [0, threads) )
    uint remains{runs - min_runs_per_thread * threads};

    std::vector<std::thread> spawned_thread;

    for (uint t{0}; t < threads; ++t)
    {
      uint thread_runs{min_runs_per_thread};
      // Distribute the 'remains' runs between threads
      if (remains > 0)
      {
        --remains;
        ++thread_runs;
      }

      // Spawn threads to run the trials in parallel
      spawned_thread.push_back(std::thread(&HexMachineMcIA::mc_task, this, t,
                                           thread_runs, selected_node_id,
                                           machine_player_id, board));
    }

    // Join threads
    for (auto& thrd : spawned_thread)
    {
      thrd.join();
    }
  }

  // Generate a bell-shape curve
  uint bell_shape(const HexBoard& board, double max, double center,
                  double sigma_sqr, double x) const
  {
    const double delta{center - x};
    const double delta_sqr{delta * delta};

    const double bell_shape{exp(-0.5 * delta_sqr / sigma_sqr)};
    const double y{max * bell_shape};

    return y;
  }
};

#endif  // HEX_MACHINE_MC_H
