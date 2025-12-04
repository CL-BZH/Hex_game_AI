#ifndef HEX_MACHINE_MCTS_H
#define HEX_MACHINE_MCTS_H

#include "hex_board.h"
#include "hex_machine_engine.h"

// MCTS Node representing a game state
struct MCTSNode
{
  uint board_row;
  uint board_column;

  MCTSNode* parent;
  std::vector<std::unique_ptr<MCTSNode>> children;

  std::atomic<uint> visits;
  AtomicDouble wins{0.0};

  uint player_id;
  std::atomic<bool> fully_expanded;
  std::vector<std::array<uint, 2>> untried_moves;

  std::mutex node_mutex;

  MCTSNode(uint row = UINT_MAX, uint col = UINT_MAX, uint player = blue_player,
           MCTSNode* parent_node = nullptr)
      : board_row(row),
        board_column(col),
        parent(parent_node),
        visits(0),
        wins(0.0),
        player_id(player),
        fully_expanded(false)
  {
  }

  bool is_root() const
  {
    return parent == nullptr;
  }

  double win_rate() const
  {
    uint v = visits.load(std::memory_order_relaxed);
    if (v == 0) return 0.0;

    double w = wins.load();
    return w / v;
  }

  double ucb1_value(double exploration_param = 1.414) const
  {
    uint v = visits.load(std::memory_order_relaxed);
    if (v == 0)
    {
      return std::numeric_limits<double>::infinity();
    }

    double w = wins.load();
    double win_rate = w / v;
    uint parent_visits =
        parent ? parent->visits.load(std::memory_order_relaxed) : 1;
    double exploration =
        exploration_param *
        std::sqrt(std::log(static_cast<double>(parent_visits)) / v);

    return win_rate + exploration;
  }

  MCTSNode* best_child_by_visits()
  {
    if (children.empty())
      throw std::runtime_error(
          "best_child_by_visit called on node with no children");

    MCTSNode* best = children[0].get();
    uint max_visits = best->visits.load(std::memory_order_relaxed);

    for (size_t i = 1; i < children.size(); ++i)
    {
      uint v = children[i]->visits.load(std::memory_order_relaxed);
      if (v > max_visits)
      {
        max_visits = v;
        best = children[i].get();
      }
    }

    return best;
  }

  MCTSNode* best_child_by_win_rate()
  {
    if (children.empty())
      throw std::runtime_error(
          "best_child_by_win_rate called on node with no children");

    MCTSNode* best = children[0].get();
    double best_win_rate = best->win_rate();

    for (size_t i = 1; i < children.size(); ++i)
    {
      double win_rate = children[i]->win_rate();
      if (win_rate > best_win_rate)
      {
        best_win_rate = win_rate;
        best = children[i].get();
      }
    }

    return best;
  }

  MCTSNode* best_child_by_ucb(double exploration_param = 1.414) const
  {
    if (children.empty())
      throw std::runtime_error(
          "best_child_by_ucb called on node with no children");

    MCTSNode* best = children[0].get();
    double max_ucb = best->ucb1_value(exploration_param);

    for (size_t i = 1; i < children.size(); ++i)
    {
      double ucb = children[i]->ucb1_value(exploration_param);
      if (ucb > max_ucb)
      {
        max_ucb = ucb;
        best = children[i].get();
      }
    }

    return best;
  }
};

// MCTS Engine for Hex
struct HexMachineMCTS : HexMachineEngine
{
  enum class SelectionStrategy
  {
    VISITS,
    WIN_RATE
  };

  HexMachineMCTS(uint board_size, uint threads = 1, bool quiet_mode = false,
                 HexStatistics* stats_ptr = nullptr,
                 SelectionStrategy strategy = SelectionStrategy::VISITS)
      : HexMachineEngine(board_size, MachineType::MCTS, threads, stats_ptr),
        exploration_param(get_exploration_constant(board_size)),
        root(nullptr),
        quiet_mode(quiet_mode),
        selection_strategy(strategy)
  {
    base_iterations = board_size * board_size * iterations_factor;
  }

  void get_position(HexBoard& board, uint& board_row, uint& board_column,
                    uint machine_player_id) override
  {
    auto start{std::chrono::high_resolution_clock::now()};

    // Get available moves
    std::vector<std::array<uint, 2>> available_pos;
    HexBoard board_tmp(board);
    board_tmp.get_all_available_position(available_pos);

    // Check for immediate winning move
    if (board.get_nb_selected_cells() >=
        2 * (board_size - 1) + machine_player_id)
    {
      if (find_winning_move(board, available_pos, board_row, board_column,
                            machine_player_id))
        return;
    }

    uint iterations = get_number_iterations(board);

    if (!quiet_mode)
    {
      std::cout << PlayersColors::color(machine_player_id) << " MCTS: Running "
                << iterations << " iterations in " << threads << " threads"
                << std::endl;
    }

    // Run MCTS
    run_mcts(board, machine_player_id, iterations);

    // Select best move
    MCTSNode* best_child = nullptr;
    if (selection_strategy == SelectionStrategy::VISITS)
      best_child = root->best_child_by_visits();
    else
      best_child = root->best_child_by_win_rate();

    board_row = best_child->board_row;
    board_column = best_child->board_column;

    auto stop{std::chrono::high_resolution_clock::now()};

    if (!quiet_mode)
    {
      print_duration(start, stop);
      print_statistics(machine_player_id);
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

    root.reset();
  }

  std::string get_name() const override
  {
    return "MCTS";
  }

private:
  static constexpr uint iterations_factor = 100;
  uint base_iterations;

  double exploration_param;
  std::unique_ptr<MCTSNode> root;
  bool quiet_mode;
  SelectionStrategy selection_strategy;

  double get_exploration_constant(uint board_size) const
  {
    return std::sqrt(2.0);
  }

  uint get_number_iterations(const HexBoard& board) const
  {
    uint total_cells = board_size * board_size;
    uint available = board.get_nb_available_cells();
    double ratio = static_cast<double>(available) / total_cells;
    return static_cast<uint>(base_iterations * (0.5 + 0.5 * ratio));
  }

  bool find_winning_move(HexBoard& board,
                         const std::vector<std::array<uint, 2>>& available,
                         uint& row, uint& col, uint player_id)
  {
    for (const auto& move : available)
    {
      HexBoard test_board(board);
      bool game_end = false;
      test_board.select(move[0], move[1], player_id, game_end);
      if (game_end)
      {
        row = move[0];
        col = move[1];
        return true;
      }
    }
    return false;
  }

  void print_statistics(uint machine_player_id)
  {
    if (!root || root->children.empty()) return;

    std::vector<MCTSNode*> children_sorted;
    for (auto& child : root->children)
    {
      children_sorted.push_back(child.get());
    }

    std::sort(children_sorted.begin(), children_sorted.end(),
              [](MCTSNode* a, MCTSNode* b)
              { return a->visits.load() > b->visits.load(); });

    if (!quiet_mode)
    {
      std::cout << PlayersColors::color(machine_player_id)
                << " Top moves:" << std::endl;

      for (size_t i = 0; i < std::min(size_t(3), children_sorted.size()); ++i)
      {
        MCTSNode* child = children_sorted[i];
        uint v = child->visits.load();
        double w = child->wins.load();
        double wr = (v > 0) ? (w / v) * 100.0 : 0.0;

        std::cout << "  Move (" << child->board_row << ","
                  << child->board_column << "): visits=" << v
                  << ", win_rate=" << std::fixed << std::setprecision(1) << wr
                  << "%" << std::endl;
      }
    }

    double root_win_rate = root->win_rate();
    int percent = static_cast<int>(root_win_rate * 100);

    std::cout << PlayersColors::color(machine_player_id)
              << " machine: Based on " << root->visits.load()
              << " simulations, I estimate " << percent
              << "% chance of winning";

    if (percent < 30)
      std::cout << " :((";
    else if (percent < 40)
      std::cout << " :(";
    else if (percent < 50)
      std::cout << " :|";

    std::cout << std::endl;
  }

  void run_mcts(HexBoard& board, uint player_id, uint iterations)
  {
    root = std::make_unique<MCTSNode>(UINT_MAX, UINT_MAX, player_id, nullptr);

    if (threads == 1)
    {
      std::mt19937 gen(rd());
      for (uint i = 0; i < iterations; ++i)
      {
        mcts_iteration(board, player_id, gen);
      }
    }
    else
    {
      run_mcts_parallel(board, player_id, iterations);
    }
  }

  void run_mcts_parallel(HexBoard& board, uint player_id, uint iterations)
  {
    uint iterations_per_thread = iterations / threads;
    uint remainder = iterations % threads;

    std::vector<std::thread> spawned_threads;
    for (uint t = 0; t < threads; ++t)
    {
      uint thread_iterations = iterations_per_thread;
      if (t < remainder) ++thread_iterations;

      spawned_threads.push_back(std::thread(&HexMachineMCTS::worker_thread,
                                            this, board, player_id,
                                            thread_iterations));
    }

    for (auto& thread : spawned_threads)
    {
      thread.join();
    }
  }

  void worker_thread(HexBoard board, uint player_id, uint iterations)
  {
    std::mt19937 gen(rd());
    for (uint i = 0; i < iterations; ++i)
    {
      mcts_iteration(board, player_id, gen);
    }
  }

  void mcts_iteration(HexBoard& board, uint player_id, std::mt19937& gen)
  {
    // 1. SELECTION PHASE
    MCTSNode* node = root.get();
    uint current_player = player_id;

    HexBoard board_copy(board);

    // Traverse down the tree using UCB1
    while (node->fully_expanded.load(std::memory_order_relaxed) &&
           !node->children.empty())
    {
      node = node->best_child_by_ucb(exploration_param);

      bool game_end = false;
      if (!board_copy.select(node->board_row, node->board_column,
                             current_player, game_end, false))
      {
        throw std::runtime_error("Failed to select move during tree traversal");
      }

      if (game_end)
      {
        backpropagate(node, player_id,
                      (current_player == player_id) ? 1.0 : 0.0);
        return;
      }

      current_player =
          (current_player == blue_player) ? red_player : blue_player;
    }

    // 2. EXPANSION PHASE
    // Try to expand if node is not fully expanded
    if (!node->fully_expanded.load(std::memory_order_relaxed))
    {
      auto expand_result = try_expand(node, board_copy, current_player, gen);

      if (expand_result.expanded)
      {
        MCTSNode* child = expand_result.child;
        bool game_end = expand_result.game_end;

        if (game_end)
        {
          backpropagate(child, player_id,
                        (current_player == player_id) ? 1.0 : 0.0);
          return;
        }

        node = child;
        current_player =
            (current_player == blue_player) ? red_player : blue_player;
      }
      // else: Race condition - another thread took the last move
      // Just simulate from current node
    }

    // 3. SIMULATION PHASE
    double result = simulate(board_copy, current_player, player_id, gen);

    // 4. BACKPROPAGATION PHASE
    backpropagate(node, player_id, result);
  }

  // Result of expansion attempt
  struct ExpandResult
  {
    bool expanded;    // True if expansion succeeded
    MCTSNode* child;  // The expanded child (if expanded=true)
    bool game_end;    // True if the expanded move ended the game
  };

  // Try to expand a node - handles race conditions gracefully
  ExpandResult try_expand(MCTSNode* node, HexBoard& board, uint player_id,
                          std::mt19937& gen)
  {
    std::lock_guard<std::mutex> lock(node->node_mutex);

    // Initialize untried moves if this is the first time
    if (node->untried_moves.empty() && node->children.empty())
    {
      HexBoard board_tmp(board);
      board_tmp.get_all_available_position(node->untried_moves);

      if (node->untried_moves.empty())
      {
        // No moves available - terminal node or board full
        node->fully_expanded.store(true, std::memory_order_relaxed);
        return {false, nullptr, false};
      }
    }

    // Check if another thread already took all moves
    if (node->untried_moves.empty())
    {
      node->fully_expanded.store(true, std::memory_order_relaxed);
      return {false, nullptr, false};
    }

    // Pick random untried move
    std::uniform_int_distribution<size_t> dist(0,
                                               node->untried_moves.size() - 1);
    size_t idx = dist(gen);
    std::array<uint, 2> move = node->untried_moves[idx];

    node->untried_moves.erase(node->untried_moves.begin() + idx);

    // Try the move
    bool game_end = false;
    bool success = board.select(move[0], move[1], player_id, game_end, false);

    if (!success)
    {
      std::stringstream err;
      err << "Failed to select available move (" << move[0] << "," << move[1]
          << ") for player " << player_id;
      throw std::runtime_error{err.str()};
    }

    // Create child node
    auto child_node =
        std::make_unique<MCTSNode>(move[0], move[1], player_id, node);
    MCTSNode* child_ptr = child_node.get();

    node->children.push_back(std::move(child_node));

    // Update expansion status
    if (node->untried_moves.empty())
    {
      node->fully_expanded.store(true, std::memory_order_relaxed);
    }

    return {true, child_ptr, game_end};
  }

  double simulate(HexBoard& board, uint current_player, uint original_player,
                  std::mt19937& gen)
  {
    // Get available moves
    std::vector<std::array<uint, 2>> available;
    HexBoard board_tmp(board);
    board_tmp.get_all_available_position(available);

    if (available.empty())
    {
      throw std::runtime_error("Started simulate with no available moves");
    }

    // Simulate on a copy
    HexBoard board_copy(board);

    while (!available.empty())
    {
      std::uniform_int_distribution<size_t> dist(0, available.size() - 1);
      size_t idx = dist(gen);
      std::array<uint, 2> move = available[idx];

      bool game_end = false;
      board_copy.select(move[0], move[1], current_player, game_end, false);

      if (game_end)
      {
        return (current_player == original_player) ? 1.0 : 0.0;
      }

      available.erase(available.begin() + idx);
      current_player =
          (current_player == blue_player) ? red_player : blue_player;
    }

    throw std::runtime_error("Simulation finished without winner");
  }

  void backpropagate(MCTSNode* node, uint original_player, double result)
  {
    while (node != nullptr)
    {
      node->visits.fetch_add(1, std::memory_order_relaxed);

      if (node->player_id == original_player)
      {
        node->wins.add(result);
      }
      else
      {
        node->wins.add(1.0 - result);
      }

      node = node->parent;
    }
  }
};

#endif  // HEX_MACHINE_MCTS_H
