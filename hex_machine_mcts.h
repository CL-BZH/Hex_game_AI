#ifndef HEX_MACHINE_MCTS_H
#define HEX_MACHINE_MCTS_H

// #include <memory>

#include "hex_board.h"
#include "hex_machine_engine.h"

// MCTS Node representing a game state
struct MCTSNode
{
  // Board position (row, column) that led to this node
  unsigned int board_row;
  unsigned int board_column;

  // Parent node
  MCTSNode* parent;

  // Children nodes
  std::vector<std::unique_ptr<MCTSNode>> children;

  // Statistics
  std::atomic<unsigned int> visits;
  AtomicDouble wins{0.0};

  // Player who made the move to reach this node
  unsigned int player_id;

  // Is this node fully expanded?
  std::atomic<bool> fully_expanded;

  // Available moves from this state
  std::vector<std::array<unsigned int, 2>> untried_moves;

  // Mutex for node expansion (still needed for single-threaded tree operations)
  std::mutex node_mutex;

  MCTSNode(unsigned int row = 0, unsigned int col = 0, unsigned int player = blue_player,
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

  // UCB1 formula for node selection
  double ucb1_value(double exploration_param = 1.414) const
  {
    unsigned int v = visits.load(std::memory_order_relaxed);
    if (v == 0)
    {
      return std::numeric_limits<double>::infinity();
    }

    double w = wins.load();
    double win_rate = w / v;
    unsigned int parent_visits = parent ? parent->visits.load(std::memory_order_relaxed) : 1;
    double exploration =
        exploration_param * std::sqrt(std::log(static_cast<double>(parent_visits)) / v);

    return win_rate + exploration;
  }

  // Check if node is terminal (game over)
  bool is_terminal() const
  {
    return children.empty() && fully_expanded.load(std::memory_order_relaxed);
  }

  // Get best child by visit count (for final move selection)
  MCTSNode* best_child_by_visits() const
  {
    if (children.empty()) return nullptr;

    const MCTSNode* best = nullptr;
    unsigned int max_visits = 0;

    for (size_t i = 0; i < children.size(); ++i)
    {
      unsigned int v = children[i]->visits.load(std::memory_order_relaxed);
      if (v > max_visits)
      {
        max_visits = v;
        best = children[i].get();
      }
    }

    return const_cast<MCTSNode*>(best);
  }

  // Get best child by UCB1 (for tree traversal)
  MCTSNode* best_child_by_ucb(double exploration_param = 1.414)
  {
    if (children.empty())
      throw std::runtime_error("best_child_by_ucb called on node with no children");

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

    visits.fetch_add(other->visits.load(std::memory_order_relaxed), std::memory_order_relaxed);
    wins.add(other->wins.load());
  }
};

// MCTS Engine for Hex
struct HexMachineMCTS : HexMachineEngine
{
  HexMachineMCTS(unsigned int board_size, unsigned int threads = 1 /*number of thread to spawn*/,
                 bool quiet_mode = false, HexStatistics* stats_ptr = nullptr)
      : HexMachineEngine(board_size, MachineType::MCTS, threads, stats_ptr),
        exploration_param(2.6),
        root(nullptr),
        quiet_mode(quiet_mode)
  {
    // Calculate iterations based on board size
    base_iterations = board_size * board_size * iterations_factor;
  }

  void get_position(HexBoard& board, unsigned int& board_row, unsigned int& board_column,
                    unsigned int machine_player_id) override
  {
    // Start chrono
    auto start{std::chrono::high_resolution_clock::now()};

    // Check for immediate winning move
    if (board.get_nb_selected_cells() >= 2 * (board_size - 1) + machine_player_id)
    {
      if (find_winning_move(board, board_row, board_column, machine_player_id)) return;
    }

    // Calculate iterations for this position
    unsigned int iterations = calculate_iterations(board);

#ifndef _NCURSES
    if (!quiet_mode)
    {
      std::cout << PlayersColors::color(machine_player_id) << " MCTS: Running " << iterations
                << " iterations in " << threads << " threads" << std::endl;
    }
#endif

    // Create main root node
    root = std::make_unique<MCTSNode>(0, 0, machine_player_id, nullptr);

    // Get available moves for the root
    std::vector<std::array<unsigned int, 2>> available_moves;
    HexBoard board_copy(board);
    board_copy.get_all_available_position(available_moves);
    root->untried_moves = available_moves;

    // Run MCTS iterations in parallel using root parallelization
    run_root_parallel_mcts(board, machine_player_id, iterations);

    // Select best move from merged tree
    MCTSNode* best = root->best_child_by_visits();

    // Stop the chrono
    auto stop{std::chrono::high_resolution_clock::now()};

    if (!quiet_mode)
    {
      // Print the time taken in number of seconds + microseconds
      print_duration(start, stop);
    }
    else if (stats != nullptr)
    {
      auto duration{std::chrono::duration_cast<std::chrono::microseconds>(stop - start)};
      std::string engine_name =
          get_name() + "_" + std::string(machine_player_id == blue_player ? "Blue" : "Red");
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
        unsigned int v = best->visits.load(std::memory_order_relaxed);
        double w = best->wins.load();
        double win_rate = (v > 0) ? (w / v) : 0.0;
        int percent = static_cast<int>(win_rate * 100);
        std::cout << PlayersColors::color(machine_player_id) << " MCTS: Selected move ("
                  << board_row << "," << board_column << ") - Win rate: " << percent << "% (" << v
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

private:
  bool quiet_mode;
  std::unique_ptr<MCTSNode> root;
  double exploration_param;
  unsigned int base_iterations;
  static constexpr unsigned int iterations_factor = 150;

  // Calculate number of iterations based on game state
  unsigned int calculate_iterations(const HexBoard& board) const
  {
    unsigned int available = board.get_nb_available_cells();
    unsigned int total = board_size * board_size;

    // More iterations early game, fewer late game
    double ratio = static_cast<double>(available) / total;

    // Bell curve centered at 0.6 of game progress
    double center = 0.6;
    double sigma_sqr = 0.1;
    double delta = ratio - center;
    double multiplier = std::exp(-0.5 * delta * delta / sigma_sqr);

    unsigned int iterations = static_cast<unsigned int>(base_iterations * multiplier);
    return std::max(100u, std::min(iterations, base_iterations));
  }

  // Check for immediate winning move
  bool find_winning_move(const HexBoard& board, unsigned int& row, unsigned int& col,
                         unsigned int player_id)
  {
    std::vector<std::array<unsigned int, 2>> available;
    HexBoard board_tmp(board);
    board_tmp.get_all_available_position(available);
    unsigned int available_size = available.size();
    unsigned int selected_cells = board_size * board_size - available_size;
    if (selected_cells <= (2 * board_size - 2))
    {
      return false;
    }

    HexBoard board_copy{board};

    for (size_t i = 0; i < available_size; ++i)
    {
      const std::array<unsigned int, 2>& pos = available[i];
      bool game_end = false;
      unsigned int node_row{pos[0]};
      unsigned int node_col{pos[1]};

      if (board_copy.select(node_row, node_col, player_id, game_end, false))
      {
        if (game_end)
        {
          row = node_row;
          col = node_col;
          return true;
        }
        board_copy.release_board_node(node_row, node_col);
      }
    }
    return false;
  }

  // Run MCTS iterations using root parallelization
  void run_root_parallel_mcts(HexBoard& board, unsigned int player_id,
                              unsigned int total_iterations)
  {
    if (threads == 1)
    {
      // Single-threaded version
      std::mt19937 local_gen(rd());
      for (unsigned int i = 0; i < total_iterations; ++i)
      {
        HexBoard board_copy(board);
        mcts_iteration_single_tree(board_copy, player_id, local_gen, *root);
      }
      return;
    }

    // Multi-threaded root parallelization
    unsigned int iters_per_thread = total_iterations / threads;
    unsigned int remainder = total_iterations % threads;

    std::vector<std::thread> thread_pool;
    std::vector<std::unique_ptr<MCTSNode>> thread_roots(threads);
    thread_pool.reserve(threads);

    try
    {
      for (unsigned int t = 0; t < threads; ++t)
      {
        unsigned int iters = iters_per_thread + (t < remainder ? 1 : 0);

        thread_pool.push_back(std::thread(
            [&, t, iters, player_id]()
            {
              // Each thread gets its own random generator with different seed
              std::mt19937 local_gen(rd() + t);

              // Create local root node for this thread
              auto local_root = std::make_unique<MCTSNode>(0, 0, player_id, nullptr);

              // Initialize with available moves
              HexBoard local_board(board);
              std::vector<std::array<unsigned int, 2>> available_moves;
              local_board.get_all_available_position(available_moves);
              local_root->untried_moves = available_moves;

              // Run iterations on local tree
              for (unsigned int i = 0; i < iters; ++i)
              {
                HexBoard board_copy(board);
                mcts_iteration_single_tree(board_copy, player_id, local_gen, *local_root);
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
      std::cerr << "Exception in run_root_parallel_mcts: " << e.what() << std::endl;
      // Join any remaining threads
      for (auto& t : thread_pool)
      {
        if (t.joinable()) t.join();
      }
      throw;
    }
  }

  // Merge trees from all threads into the main root
  void merge_thread_trees(const std::vector<std::unique_ptr<MCTSNode>>& thread_roots)
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
            other_child->board_row, other_child->board_column, other_child->player_id, main_node);

        new_child->visits.store(other_child->visits.load(std::memory_order_relaxed));
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
  void mcts_iteration_single_tree(HexBoard& board, unsigned int player_id, std::mt19937& gen,
                                  MCTSNode& tree_root)
  {
    // 1. Selection - traverse tree using UCB1
    MCTSNode* node = &tree_root;
    unsigned int current_player = player_id;
    HexBoard board_copy(board);

    while (true)
    {
      if (!node->untried_moves.empty())
      {
        // Expansion phase
        node = expand(node, board_copy, current_player, gen);
        if (node == nullptr) throw std::runtime_error("Failed to expand with an untried node");
        break;
      }
      else if (!node->children.empty())
      {
        // Selection phase
        node = node->best_child_by_ucb(exploration_param);

        // Apply move to board
        bool game_end = false;
        if (!board_copy.select(node->board_row, node->board_column, current_player, game_end,
                               false))
          throw std::runtime_error("Failed to select the node returned by best_child_by_ucb");

        if (game_end)
        {
          // Terminal node - backpropagate win
          backpropagate(node, current_player, 1.0);
          return;
        }

        current_player = (current_player == blue_player) ? red_player : blue_player;
      }
      else
      {
        // Leaf node reached
        throw std::runtime_error("mcts_iteration called with no expansion possible");
      }
    }

    // 2. Simulation - random playout
    double result = simulate(board_copy, current_player, player_id, gen);

    // 3. Backpropagation
    backpropagate(node, player_id, result);
  }

  // Expand node by trying one untried move
  // expand receives a copy of the board
  MCTSNode* expand(MCTSNode* node, HexBoard& board, unsigned int player_id, std::mt19937& gen)
  {
    std::lock_guard<std::mutex> lock(node->node_mutex);

    // Pick random untried move
    std::uniform_int_distribution<size_t> dist(0, node->untried_moves.size() - 1);
    size_t idx = dist(gen);
    std::array<unsigned int, 2> move = node->untried_moves[idx];
    node->untried_moves.erase(node->untried_moves.begin() + idx);

    // Try the move - THIS SHOULD NEVER FAIL!
    bool game_end = false;
    bool success = board.select(move[0], move[1], player_id, game_end, false);

    if (!success)
    {
      // This is a serious error - throw exception
      std::stringstream err;
      err << "Failed to select available move (" << move[0] << "," << move[1] << ") for player "
          << player_id;
      throw std::runtime_error{err.str()};
    }

    // Continue with normal expansion...
    std::unique_ptr<MCTSNode> child = std::make_unique<MCTSNode>(move[0], move[1], player_id, node);

    if (!game_end)
    {
      HexBoard board_copy(board);
      std::vector<std::array<unsigned int, 2>> child_moves;
      board_copy.get_all_available_position(child_moves);
      child->untried_moves = child_moves;
    }
    else
    {
      child->fully_expanded.store(true, std::memory_order_relaxed);
    }

    MCTSNode* child_ptr = child.get();
    node->children.push_back(std::move(child));

    if (node->untried_moves.empty()) node->fully_expanded.store(true, std::memory_order_relaxed);

    return child_ptr;
  }

  // Simulate random playout from current position
  double simulate(HexBoard& board, unsigned int current_player, unsigned int original_player,
                  std::mt19937& gen)
  {
    std::vector<std::array<unsigned int, 2>> available;
    HexBoard board_tmp(board);
    board_tmp.get_all_available_position(available);

    if (available.empty())
    {
      std::stringstream err;
      err << "Started simulate with available empty! ";
      err << "nb_available_cells() = " << board.get_nb_available_cells() << std::endl;
      throw std::runtime_error{err.str()};
    }

    HexBoard board_copy(board);

    // Quick random playout
    while (!available.empty())
    {
      std::uniform_int_distribution<size_t> dist(0, available.size() - 1);
      size_t idx = dist(gen);
      std::array<unsigned int, 2> move = available[idx];

      bool game_end = false;
      board_copy.select(move[0], move[1], current_player, game_end, false);

      if (game_end)
      {
        // Game over - calculate quality-based reward
        double quality = 0.0;
        board_copy.has_won(current_player, nullptr, &quality);

        // Return win value based on quality
        if (current_player == original_player)
          return 1;
        else
          return 0;
      }

      available.erase(available.begin() + idx);
      current_player = (current_player == blue_player) ? red_player : blue_player;
    }

    // Shouldn't reach here since draw is impossible
    std::stringstream err;
    err << "Impossible case: simulation finished with a draw" << std::endl;
    throw std::runtime_error{err.str()};
  }

  // Backpropagate result up the tree
  void backpropagate(MCTSNode* node, unsigned int original_player, double result)
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
