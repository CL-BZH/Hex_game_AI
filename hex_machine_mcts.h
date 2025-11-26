#ifndef HEX_MACHINE_MCTS_H
#define HEX_MACHINE_MCTS_H

#include "hex_board.h"
#include "hex_machine_engine.h"

// MCTS Node representing a game state
struct MCTSNode
{
  // Board position (row, column) that led to this node (i.e. The board position
  // (row, column) that was played to reach this game state from the parent
  // node.
  // - For the root node: Represents "no move" (sentinel values UINT_MAX)
  // - For child nodes: Represents the actual move (row, column) that was played
  // to transition from the parent's game state to this node's game state
  uint board_row;
  uint board_column;

  // Parent node
  MCTSNode* parent;

  // Children nodes
  std::vector<std::unique_ptr<MCTSNode>> children;

  // Statistics
  std::atomic<uint> visits;
  AtomicDouble wins{0.0};

  // Player who made the move to reach this node
  uint player_id;

  // Is this node fully expanded?
  std::atomic<bool> fully_expanded;

  // Available moves from this state
  std::vector<std::array<uint, 2>> untried_moves;

  // Mutex for node expansion
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

  // Copy constructor
  MCTSNode(const MCTSNode& other)
      : board_row(other.board_row),
        board_column(other.board_column),
        parent(other.parent),  // Note: careful with parent pointer in copies
        visits(other.visits.load(std::memory_order_relaxed)),
        wins(other.wins.load()),
        player_id(other.player_id),
        fully_expanded(other.fully_expanded.load(std::memory_order_relaxed)),
        untried_moves(other.untried_moves)
  {
    // Note: children are NOT copied - this creates a shallow copy of the node
    // only The mutex is not copied (each node gets its own mutex)
  }

  // Copy assignment operator
  MCTSNode& operator=(const MCTSNode& other)
  {
    if (this != &other)
    {
      board_row = other.board_row;
      board_column = other.board_column;
      parent = other.parent;
      visits.store(other.visits.load(std::memory_order_relaxed));
      wins.store(other.wins.load());
      player_id = other.player_id;
      fully_expanded.store(
          other.fully_expanded.load(std::memory_order_relaxed));
      untried_moves = other.untried_moves;
      // children are NOT copied
    }
    return *this;
  }

  // Tell if a node represent the current state of the board from where
  // the algorithm is run.
  bool is_root() const
  {
    return parent == nullptr;
  }

  double win_rate() const
  {
    uint v = visits.load(std::memory_order_relaxed);
    if (v == 0)
    {
      return std::numeric_limits<double>::infinity();
    }

    double w = wins.load();
    return w / v;
  }

  // UCB1 formula for node selection
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

  // Get the best child by visit count (for final move selection)
  MCTSNode* best_child_by_visits()
  {
    if (!is_root())
      throw std::runtime_error(
          "best_child_by_ucb called on node that is not the root");

    if (children.empty())
      throw std::runtime_error(
          "best_child_by_visit called on node with no children");

    const MCTSNode* best = nullptr;
    uint max_visits = 0;

    for (size_t i = 0; i < children.size(); ++i)
    {
      uint v = children[i]->visits.load(std::memory_order_relaxed);
      if (v > max_visits)
      {
        max_visits = v;
        best = children[i].get();
      }
    }

    return const_cast<MCTSNode*>(best);
  }

  // Get the best child according to the win-to-visit ratio (for final move
  // selection)
  MCTSNode* best_child_by_win_rate()
  {
    if (!is_root())
      throw std::runtime_error(
          "best_child_by_ucb called on node that is not the root");

    if (children.empty())
      throw std::runtime_error(
          "best_child_by_win_rate called on node with no children");

    const MCTSNode* best = children[0].get();  // Start with first child
    double best_win_rate = best->win_rate();

    for (size_t i = 0; i < children.size(); ++i)
    {
      double win_rate = children[i]->win_rate();
      if (win_rate > best_win_rate)
      {
        best_win_rate = win_rate;
        best = children[i].get();
      }
    }

    return const_cast<MCTSNode*>(best);
  }

  // Get the best child by UCB1 (for tree traversal)
  MCTSNode* best_child_by_ucb(double exploration_param = 1.414) const
  {
    if (!is_root())
      throw std::runtime_error(
          "best_child_by_ucb called on node that is not the root");

    if (children.empty())
      throw std::runtime_error(
          "best_child_by_ucb called on node with no children");

    MCTSNode* best = children[0].get();  // Start with first child
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

    return best;  // Guaranteed non-null since children not empty
  }

  // Merge statistics from another node (for root parallelization)
  void merge_from(const MCTSNode* other)
  {
    if (other->board_row != board_row || other->board_column != board_column)
    {
      return;  // Can only merge identical nodes
    }

    visits.fetch_add(other->visits.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    wins.add(other->wins.load());
  }
};

// MCTS Engine for Hex
struct HexMachineMCTS : HexMachineEngine
{
  enum class SelectionStrategy
  {
    VISITS,   // Select by visit count (default)
    WIN_RATE  // Select by win rate
  };

  HexMachineMCTS(uint board_size,
                 uint threads = 1 /*number of thread to spawn*/,
                 bool quiet_mode = false, HexStatistics* stats_ptr = nullptr,
                 SelectionStrategy strategy = SelectionStrategy::VISITS)
      : HexMachineEngine(board_size, MachineType::MCTS, threads, stats_ptr),
        exploration_param(get_exploration_constant(board_size)),
        root(nullptr),
        quiet_mode(quiet_mode),
        selection_strategy(strategy)
  {
    // Calculate iterations based on board size
    base_iterations = board_size * board_size * iterations_factor;
  }

  void get_position(HexBoard& board, uint& board_row, uint& board_column,
                    uint machine_player_id) override
  {
    // Start chrono
    auto start{std::chrono::high_resolution_clock::now()};

    // Get available moves for the root
    std::vector<std::array<uint, 2>> available_pos;
    HexBoard board_copy(board);
    board_copy.get_all_available_position(available_pos);

    // Check for immediate winning move
    if (board.get_nb_selected_cells() >=
        2 * (board_size - 1) + machine_player_id)
    {
      if (find_winning_move(board, available_pos, board_row, board_column,
                            machine_player_id))
        return;
    }

    // Calculate iterations for this position
    uint iterations = calculate_iterations(board);

#ifndef _NCURSES
    if (!quiet_mode)
    {
      std::cout << PlayersColors::color(machine_player_id) << " MCTS: Running "
                << iterations << " iterations in " << threads << " threads"
                << std::endl;
    }
#endif

    // Create main root node (represent the current board)
    root = std::make_unique<MCTSNode>(UINT_MAX, UINT_MAX, machine_player_id,
                                      nullptr);

    root->untried_moves = available_pos;

    // Run MCTS iterations in parallel using root parallelization
    run_root_parallel_mcts(board, machine_player_id, iterations);

    // Select best move from merged tree using the configured strategy
    MCTSNode* best = select_best_child();

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

    if (best == nullptr)
    {
      throw std::runtime_error("No best move");
    }
    else
    {
      board_row = best->board_row;
      board_column = best->board_column;

#ifndef _NCURSES
      if (!quiet_mode)
      {
        uint v = best->visits.load(std::memory_order_relaxed);
        double w = best->wins.load();
        double win_rate = (v > 0) ? (w / v) : 0.0;
        int percent = static_cast<int>(win_rate * 100);
        std::cout << PlayersColors::color(machine_player_id)
                  << " MCTS: Selected move (" << board_row << ","
                  << board_column << ") - Win rate: " << percent << "% (" << v
                  << " visits)" << std::endl;
      }
#endif
    }

    // Clean up for next move
    root.reset();
  }

  std::string get_name() const override
  {
    return "MCTS";
  }

  // Method to change strategy at runtime
  void set_selection_strategy(SelectionStrategy strategy)
  {
    selection_strategy = strategy;
  }

private:
  // Disable print out
  bool quiet_mode;

  // Current board
  std::unique_ptr<MCTSNode> root;

  // Exploration parameter (c) for UCB1
  double exploration_param;

  // Iterations based on board size and iteration_factor
  uint base_iterations;

  static constexpr uint iterations_factor = 200;

  // Selection strategy for final move choice
  SelectionStrategy selection_strategy;

  // Helper method to select best child based on configured strategy
  MCTSNode* select_best_child()
  {
    switch (selection_strategy)
    {
      case SelectionStrategy::VISITS:
        return root->best_child_by_visits();
      case SelectionStrategy::WIN_RATE:
        return root->best_child_by_win_rate();
      default:
        throw std::runtime_error("Unknown selection strategy");
    }
  }

  double get_exploration_constant(uint board_size)
  {
    // For example, for board_size equal 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // this function returns 2.64 2.46 2.28 2.1  1.92 1.74 1.56 1.38 1.2  1.02
    // 0.84 respectively
    double c = 3.0 - (board_size - 3) * 0.18;
    if (!quiet_mode)
    {
      std::cout << "Exploration constante for UCB1: " << c << std::endl;
    }
    return c;
  }

  // Calculate number of iterations based on game state
  uint calculate_iterations(const HexBoard& board) const
  {
    uint available = board.get_nb_available_cells();
    uint total = board_size * board_size;

    // More iterations early game, fewer late game
    double ratio = static_cast<double>(available) / total;

    // Bell curve centered at 0.6 of game progress
    double center = 0.6;
    double sigma_sqr = 0.1;
    double delta = ratio - center;
    double multiplier = std::exp(-0.5 * delta * delta / sigma_sqr);

    uint iterations = static_cast<uint>(base_iterations * multiplier);
    return std::max(100u, std::min(iterations, base_iterations));
  }

  // Check for immediate winning move
  bool find_winning_move(const HexBoard& board,
                         std::vector<std::array<uint, 2>> available_pos,
                         uint& row, uint& col, uint player_id)
  {
    uint available_size = available_pos.size();
    HexBoard board_copy{board};

    // Since the same board will be used multiple times, we have to
    // backup the current state of the player's UnionFind object in
    // order to restore it after each iteration
    board_copy.save_union_find_state(player_id);

    for (size_t i = 0; i < available_size; ++i)
    {
      const std::array<uint, 2>& pos = available_pos[i];
      bool game_end = false;
      uint node_row{pos[0]};
      uint node_col{pos[1]};

      if (board_copy.select(node_row, node_col, player_id, game_end))
      {
        if (game_end)
        {
          row = node_row;
          col = node_col;
          return true;
        }
        board_copy.release_board_node(node_row, node_col);
        board_copy.restore_union_find_state(player_id);
      }
    }
    return false;
  }

  // Run MCTS iterations using root parallelization
  void run_root_parallel_mcts(HexBoard& board, uint player_id,
                              uint total_iterations)
  {
#if _DEBUG
    if ((root->board_row != UINT_MAX) || (root->board_column != UINT_MAX))
      throw std::runtime_error("root node is incorrectly set");
#endif

    if (threads == 1)
    {
      // Single-threaded version
      std::mt19937 local_gen(rd());
      for (uint i = 0; i < total_iterations; ++i)
      {
        mcts_iteration_single_tree(board, player_id, local_gen, *root);
      }
      return;
    }

    // Multi-threaded root parallelization
    uint iters_per_thread = total_iterations / threads;
    uint remainder = total_iterations % threads;

    std::vector<std::thread> thread_pool;
    std::vector<std::unique_ptr<MCTSNode>> thread_roots(threads);
    thread_pool.reserve(threads);

    try
    {
      for (uint t = 0; t < threads; ++t)
      {
        uint iters = iters_per_thread + (t < remainder ? 1 : 0);

        thread_pool.push_back(std::thread(
            [&, t, iters, player_id]()
            {
              // Each thread gets its own random generator with different seed
              std::mt19937 local_gen(rd() + t);

              // Create local root node for this thread using the copy
              // constructor
              auto local_root = std::make_unique<MCTSNode>(*root);

              // Run iterations on local tree
              for (uint i = 0; i < iters; ++i)
              {
                mcts_iteration_single_tree(board, player_id, local_gen,
                                           *local_root);
              }

              // Store the local root
              thread_roots[t] = std::move(local_root);
            }));
      }

      // Wait for all threads to complete
      for (auto& t : thread_pool)
      {
        t.join();
      }

      // Merge all thread results into the main root
      merge_thread_trees(thread_roots);
    }
    catch (const std::exception& e)
    {
      std::cerr << "Exception in run_root_parallel_mcts: " << e.what()
                << std::endl;
      // Join any remaining threads
      for (auto& t : thread_pool)
      {
        if (t.joinable()) t.join();
      }
      throw;
    }
  }

  // Merge trees from all threads into the main root
  void merge_thread_trees(
      const std::vector<std::unique_ptr<MCTSNode>>& thread_roots)
  {
    for (const auto& thread_root : thread_roots)
    {
      if (!thread_root) continue;
      merge_nodes(root.get(), thread_root.get());
    }
  }

  // Recursively merge two nodes and their children
  void merge_nodes(MCTSNode* main_node, const MCTSNode* other_node)
  {
    if (!main_node || !other_node) return;

    // Merge this node's statistics
    main_node->merge_from(other_node);

    // Merge children recursively
    for (const auto& other_child : other_node->children)
    {
      // Find matching child in main node
      MCTSNode* matching_child = nullptr;
      for (const auto& main_child : main_node->children)
      {
        if (main_child->board_row == other_child->board_row &&
            main_child->board_column == other_child->board_column)
        {
          matching_child = main_child.get();
          break;
        }
      }

      if (matching_child)
      {
        // Child exists, merge recursively
        merge_nodes(matching_child, other_child.get());
      }
      else
      {
        // Child doesn't exist in main tree, add it
        auto new_child = std::make_unique<MCTSNode>(
            other_child->board_row, other_child->board_column,
            other_child->player_id, main_node);

        new_child->visits.store(
            other_child->visits.load(std::memory_order_relaxed));
        new_child->wins.store(other_child->wins.load());
        new_child->fully_expanded.store(
            other_child->fully_expanded.load(std::memory_order_relaxed));
        new_child->untried_moves = other_child->untried_moves;

        // Add to main node's children
        main_node->children.push_back(std::move(new_child));
      }
    }
  }

  // Single MCTS iteration on a single tree (thread-safe)
  void mcts_iteration_single_tree(HexBoard& board, uint player_id,
                                  std::mt19937& gen, MCTSNode& tree_root)
  {
    // 1. Selection - traverse tree using UCB1
    MCTSNode* node = &tree_root;
    uint current_player = player_id;
    HexBoard board_copy(board);

    if (!node->fully_expanded.load(std::memory_order_relaxed))
    {
      // Node might have untried moves
      if (!node->untried_moves.empty())
      {
        // Expand untried move
        auto [child, game_end] = expand(node, board_copy, current_player, gen);
        if (game_end)
        {
          backpropagate(child, player_id, 1.0);
          return;
        }
        node = child;
      }
      else
      {
        // INCONSISTENT STATE: Should never happen!
        std::stringstream err;
        err << "MCTS node inconsistency: "
            << "fully_expanded==false but untried_moves is empty. "
            << "Node at (" << node->board_row << "," << node->board_column
            << ")" << std::endl;
        throw std::runtime_error(err.str());
      }
    }
    else if (!node->children.empty())
    {
#if _DEBUG
      if ((node->board_row != UINT_MAX) || (node->board_column != UINT_MAX))
        throw std::runtime_error("node is not root");
#endif
      // Selection phase
      node = node->best_child_by_ucb(exploration_param);

      // Apply move to board
      bool game_end = false;
      if (!board_copy.select(node->board_row, node->board_column,
                             current_player, game_end, false))
        throw std::runtime_error(
            "Failed to select the node returned by best_child_by_ucb");

      if (game_end)
      {
        // Terminal node - backpropagate win
        backpropagate(node, current_player, 1.0);
        return;
      }
    }
    else
    {
      // Leaf node reached
      throw std::runtime_error(
          "mcts_iteration called with no expansion possible");
    }

    // 2. Simulation - random playout
    current_player = (current_player == blue_player) ? red_player : blue_player;
    double result = simulate(board_copy, current_player, player_id, gen);

    // 3. Backpropagation
    backpropagate(node, player_id, result);
  }

  // Expand node by trying one untried move
  // expand receives a copy of the board
  // Returns pair<child_node, game_end_status>
  std::pair<MCTSNode*, bool> expand(MCTSNode* node, HexBoard& board,
                                    uint player_id, std::mt19937& gen)
  {
    std::lock_guard<std::mutex> lock(node->node_mutex);

    if (!node->is_root())
    {
      throw std::runtime_error("expand called on a non-root node");
    }

    if (node->untried_moves.empty())
      throw std::runtime_error("No moves available for expansion");

    // Pick random untried move
    std::uniform_int_distribution<size_t> dist(0,
                                               node->untried_moves.size() - 1);
    size_t idx = dist(gen);
    std::array<uint, 2> move = node->untried_moves[idx];

    // Remove the move from untried moves
    node->untried_moves.erase(node->untried_moves.begin() + idx);

    // Try the move
    bool game_end = false;
    bool success = board.select(move[0], move[1], player_id, game_end);

    if (!success)
    {
      std::stringstream err;
      err << "Failed to select available move (" << move[0] << "," << move[1]
          << ") for player " << player_id << " on board ID: " << board.get_id()
          << std::endl;
      throw std::runtime_error{err.str()};
    }

    // Selection succeed: in the copy of the board passed to expand()
    // the position (move[0], move[1]) is now marked as belonging to the
    // player with id 'player_id'.

    // Create and add child node
    auto child_node =
        std::make_unique<MCTSNode>(move[0], move[1], player_id, node);
    MCTSNode* child_ptr = child_node.get();

    // The child node is now an "evaluated child" of the root node
    node->children.push_back(std::move(child_node));

    // Update root node expansion status
    node->fully_expanded.store(node->untried_moves.empty(),
                               std::memory_order_relaxed);

    return std::make_pair(child_ptr, game_end);
  }

  // Simulate random playout from current position
  double simulate(HexBoard& board, uint current_player, uint original_player,
                  std::mt19937& gen)
  {
    std::vector<std::array<uint, 2>> available;
    HexBoard board_tmp(board);
    board_tmp.get_all_available_position(available);

    if (available.empty())
    {
      std::stringstream err;
      err << "Started simulate with available empty! ";
      err << "nb_available_cells() = " << board.get_nb_available_cells()
          << std::endl;
      throw std::runtime_error{err.str()};
    }

    HexBoard board_copy(board);

    // Quick random playout
    while (!available.empty())
    {
      std::uniform_int_distribution<size_t> dist(0, available.size() - 1);
      size_t idx = dist(gen);
      std::array<uint, 2> move = available[idx];

      bool game_end = false;
      board_copy.select(move[0], move[1], current_player, game_end);

      if (game_end)
      {
        // Game over - calculate quality-based reward
        double quality = 0.0;
        bool has_won = board_copy.has_won(current_player, nullptr, &quality);

        if (!has_won)
        {
          std::stringstream err;
          err << "game_end is true but has_won is false" << std::endl;
          throw std::runtime_error{err.str()};
        }

        // Return win value based on quality
        if (current_player == original_player)
          return 1;
        else
          return 0;
      }

      available.erase(available.begin() + idx);
      current_player =
          (current_player == blue_player) ? red_player : blue_player;
    }

    // Shouldn't reach here since draw is impossible
    std::stringstream err;
    err << "Impossible case: simulation finished with a draw" << std::endl;
    throw std::runtime_error{err.str()};
  }

  // Backpropagate result up the tree
  void backpropagate(MCTSNode* node, uint original_player, double result)
  {
    while (node != nullptr)
    {
      node->visits.fetch_add(1, std::memory_order_relaxed);

      if (node->player_id == original_player)
        node->wins.add(result);
      else
        node->wins.add(1.0 - result);

      node = node->parent;
    }
  }
};

#endif  // HEX_MACHINE_MCTS_H
