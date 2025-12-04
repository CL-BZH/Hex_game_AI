"""
================================================================================
HEX ALPHAZERO - COMPLETE EDUCATIONAL IMPLEMENTATION
================================================================================

This file implements 3 Monte Carlo approaches for playing Hex, from simplest
to most sophisticated:

┌─────────────────┬──────────────┬────────────────┬─────────────┐
│   Algorithm     │  Tree Depth  │   Simulation   │  Selection  │
├─────────────────┼──────────────┼────────────────┼─────────────┤
│ 1. Pure MC      │  None        │  Random        │  None       │
│ 2. Naive MCTS   │  Deep (grows)│  Random        │  UCB1       │
│ 3. AlphaZero    │  Deep (grows)│  NN (no sim!)  │  PUCT+NN    │
└─────────────────┴──────────────┴────────────────┴─────────────┘

EDUCATIONAL PROGRESSION:
========================
Pure MC →  Naive MCTS → AlphaZero
               ↓          ↓
          Adds UCB1   Adds NN

TERMINOLOGY CLARIFICATION:
==========================
- "Naive" = Random simulation policy (not smart)
- "MCTS" = Monte Carlo Tree Search (grows deep tree)

FILE STRUCTURE:
===============
1. Hex Game Implementation
2. Neural Network (ResNet)
3. MCTS Node
4. Pure Monte Carlo
5. Naive MCTS
6. AlphaZero MCTS
7. Training Loop
8. Player Classes
9. Visualization & Logging
10. Comparison Utilities

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import json
from datetime import datetime
import time

# =============================================================================
# HEX GAME CONSTANTS AND GEOMETRY
# =============================================================================

# Hexagon rendering constants
HEX_SIDE_LENGTH = 1.0
HEX_VERTICAL_SPACING = np.sqrt(3) / 2  # ≈ 0.866
HEX_HORIZONTAL_OFFSET = 0.5

HEX_VERTICES = np.array([
    [0, 0.57735], [0.5, 0.288675], [0.5, -0.288675],
    [0, -0.57735], [-0.5, -0.288675], [-0.5, 0.288675]
])

# Player constants
PLAYER_BLUE = 1    # Connects left to right
PLAYER_RED = -1    # Connects top to bottom

# =============================================================================
# 1. HEX GAME IMPLEMENTATION
# =============================================================================

class HexGame:
    """Hex game with standard rules - see previous implementation"""

    def __init__(self, board_size=5):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = PLAYER_BLUE
        self.game_over = False
        self.winner = None
        self.empty_board = True

    def clone(self):
        new_game = HexGame(self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.empty_board = self.empty_board
        return new_game

    def get_legal_moves(self):
        if self.game_over:
            return []
        return [(i, j) for i in range(self.board_size)
                for j in range(self.board_size) if self.board[i, j] == 0]

    def make_move(self, move):
        row, col = move
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid move {move}")

        self.board[row, col] = self.current_player

        if self.check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player

        if self.empty_board == True:
            self.empty_board = False

        self.current_player = -self.current_player

    def is_empty(self):
        return self.empty_board

    def check_win(self, player):
        if player == PLAYER_BLUE:
            start = [(i, 0) for i in range(self.board_size)]
            end = [(i, self.board_size-1) for i in range(self.board_size)]
        else:
            start = [(0, j) for j in range(self.board_size)]
            end = [(self.board_size-1, j) for j in range(self.board_size)]
        return self._path_exists(player, start, end)

    def _path_exists(self, player, start_cells, end_cells):
        visited = set()
        queue = deque()
        for cell in start_cells:
            if self.board[cell[0], cell[1]] == player:
                queue.append(cell)
                visited.add(cell)

        while queue:
            row, col = queue.popleft()
            if (row, col) in end_cells:
                return True
            for nr, nc in self._get_neighbors(row, col):
                if (nr, nc) not in visited and self.board[nr, nc] == player:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _get_neighbors(self, row, col):
        neighbors = [
            (row-1, col), (row-1, col+1),
            (row, col-1), (row, col+1),
            (row+1, col-1), (row+1, col)
        ]
        return [(r, c) for r, c in neighbors
                if 0 <= r < self.board_size and 0 <= c < self.board_size]

    def get_board_representation(self):
        board_3d = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        board_3d[0] = (self.board == self.current_player).astype(np.float32)
        board_3d[1] = (self.board == -self.current_player).astype(np.float32)
        board_3d[2] = (self.board == 0).astype(np.float32)
        return board_3d

    def move_to_index(self, move):
        return move[0] * self.board_size + move[1]

    def index_to_move(self, index):
        return (index // self.board_size, index % self.board_size)

# =============================================================================
# 2. NEURAL NETWORK
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = F.relu(out)
        return out

class HexNetwork(nn.Module):
    def __init__(self, board_size=5, num_res_blocks=5, num_channels=64):
        super().__init__()
        self.board_size = board_size
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

# =============================================================================
# 3. MCTS NODE
# =============================================================================

class MCTSNode:
    """
    Node in MCTS tree

    Used by both Naive MCTS and AlphaZero MCTS.
    """

    def __init__(self, game_state, parent=None, move=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __repr__(self):
        # Print action, visits, wins, player, and root perspective
        return f"Node(S: {self.game_state}, A: {self.move}, V: {self.visit_count}, "\
            f"W: {self.value_sum:.1f}"

# =============================================================================
# 4. PURE MONTE CARLO
# =============================================================================

class PureMonteCarlo:
    """
    Pure Monte Carlo - The Simplest Approach
    =========================================

    NO TREE AT ALL!

    Algorithm:
    ----------
    For each legal move:
        1. Make the move
        2. Run N random games to completion
        3. Count wins
        4. Calculate win rate = wins / N

    Return move with highest win rate

    Characteristics:
    ----------------
    ✗ No tree
    ✗ No learning from previous simulations
    ✗ Equal budget for all moves (inefficient!)
    ✓ Simple to implement
    ✓ Good baseline

    Example (5 moves, 1000 sims total):
    ------------------------------------
    Move A: 200 sims → 90 wins  (45%)
    Move B: 200 sims → 120 wins (60%) ← Best!
    Move C: 200 sims → 80 wins  (40%)
    Move D: 200 sims → 100 wins (50%)
    Move E: 200 sims → 70 wins  (35%)

    Problem: Wasted 200 sims on bad moves A, C, E!
    """

    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations

    def get_move(self, game_state):
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        move_stats = {}

        for move in legal_moves:
            wins = 0
            for _ in range(self.num_simulations):
                sim_game = game_state.clone()
                sim_game.make_move(move)
                result = self._random_playout(sim_game, game_state.current_player)
                if result == 1.0:
                    wins += 1

            move_stats[move] = wins / self.num_simulations

        return max(move_stats.keys(), key=lambda m: move_stats[m])

    def _random_playout(self, game_state, original_player):
        while not game_state.game_over:
            legal_moves = game_state.get_legal_moves()
            if not legal_moves:
                break
            game_state.make_move(random.choice(legal_moves))

        if game_state.winner == original_player:
            return 1.0
        else:
            return -1.0

# =============================================================================
# 4. NAIVE MCTS
# =============================================================================
class NaiveMCTS:
    """
    Naive MCTS - Traditional MCTS with Random Rollouts
    ===================================================

    TREE GROWS DEEP!

    Tree Structure (grows over time):
    ----------------------------------
                         Root
                       /  |   \
                      /   |    \
                Child1  Child2  Child3
                   /\      /\
                 /   \   /   \
                 Grandchildren...
                       |
                Great-grandchildren...

    Four Phases (repeated N times):
    --------------------------------

    1. SELECTION
       Navigate tree using UCB1, choosing best child at each level
       until reaching a leaf node

       Example path: Root → Child2 → Grandchild3 → Leaf

    2. EXPANSION
       Add ONE new child to the leaf
       (Or use leaf directly if it's terminal)

    3. SIMULATION
       Play randomly from the new node to game end
       (This is the "naive" part - random policy)

    4. BACKPROPAGATION
       Update ALL nodes in the path from leaf to root
       Flip value at each level (alternating players)

    Why "Naive"?
    ------------
    The "naive" part refers to the SIMULATION POLICY (random play).
    Smarter simulations → stronger MCTS
    AlphaZero eliminates simulation entirely using neural network!

    """

    def __init__(self, num_simulations=1000, exploration_constant=1.4,
                 log_tree=False, log_file=None):
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.log_tree = log_tree
        self.log_file = log_file
        self.tree_log = []

    def search(self, game_state):
        """
        Main MCTS search with deep tree building

        Returns:
            (move_probs, root_value): Probabilities and position value
        """
        root = MCTSNode(game_state.clone())

        if self.log_tree:
            self._log_state(root, 0)

        # Run simulations - tree grows incrementally
        for sim in range(1, self.num_simulations + 1):
            node = root
            search_path = [node]

            #===================================================
            # PHASE 1: SELECTION
            # Navigate down the tree using UCB1
            #===================================================
            while node.is_expanded:
                move, node = self._select_child(node)
                if node is None:
                    raise ValueError(f"node {node} has no child!")
                search_path.append(node)

                # If we reached a terminal node, break
                if node.game_state.game_over:
                    break

            #===================================================
            # PHASE 2: EXPANSION
            # Add ONE new node to tree (if not terminal)
            #===================================================
            leaf_node = node

            # If game is over at this leaf, no expansion needed
            if not leaf_node.game_state.game_over:
                # Expand: add one child
                child = self._expand(leaf_node)
                if child is not None:
                    search_path.append(child)
                    leaf_node = child  # Continue from the new child
                else:
                    raise ValueError(f"leaf node {leaf_node} has no child!")

            #===================================================
            # PHASE 3: SIMULATION
            # Random playout from leaf node
            #===================================================
            value = self._random_simulation(leaf_node.game_state.clone(),
                                            leaf_node.game_state.current_player)

            #===================================================
            # PHASE 4: BACKPROPAGATION
            # Update all nodes in path
            #===================================================
            self._backpropagate(search_path, value)

            if self.log_tree and (sim <= 10 or sim % 50 == 0):
                self._log_state(root, sim)

        if self.log_tree and self.log_file:
            self._save_log(game_state.board_size)

        return self._get_move_probs(root)

    def _select_child(self, node):
        """
        UCB1 selection: Choose child with highest UCB1 score

        UCB1 formula: Q + c * sqrt(ln(N_parent) / N_child)
        where Q = win rate, c = exploration constant
        """
        if node.game_state.game_over:
            return None, node

        best_score = -float('inf')
        best_move, best_child = None, None

        # Calculate UCB1 for each child
        for move, child in node.children.items():
            # If child has never been visited, give it maximum priority
            if child.visit_count == 0:
                ucb1_score = float('inf')
                return move, child
            else:
                # Q: exploitation (win rate from current player's perspective)
                q_value = child.value()

                # U: exploration term
                # Use parent's visit count, not the node's own
                parent_visits = node.visit_count if node.visit_count > 0 else 1
                u_value = (self.exploration_constant *
                          np.sqrt(np.log(parent_visits) / (child.visit_count + 1e-8)))

                ucb1_score = q_value + u_value

            if ucb1_score > best_score:
                best_score = ucb1_score
                best_move, best_child = move, child

        return best_move, best_child

    def _expand(self, node):
        """
        Expand a leaf node by adding ONE random child

        This is where tree GROWS incrementally!
        Only add one child per simulation.
        """
        if node.game_state.game_over:
            return node  # Terminal node, no expansion possible

        # Get legal moves that haven't been expanded yet
        legal_moves = node.game_state.get_legal_moves()

        # Filter out moves that already have children
        unexpanded_moves = [move for move in legal_moves
                           if move not in node.children]

        if not unexpanded_moves:
            # All moves have been expanded
            node.is_expanded = True
            return node

        # Choose one random unexpanded move to expand
        move = random.choice(unexpanded_moves)

        # Create new child state
        child_state = node.game_state.clone()
        child_state.make_move(move)

        # Create child node
        # Prior is uniform for Naive MCTS (unlike AlphaZero which uses NN)
        child = MCTSNode(child_state, parent=node, move=move, prior=1.0)
        node.children[move] = child

        # Mark node as expanded if all moves have children
        if len(node.children) == len(legal_moves):
            node.is_expanded = True

        return child

    def _random_simulation(self, game_state, leaf_node_next_player):
        """
        Random playout from current position to terminal state

        Returns:
            value: +1 if leaf node player (at start of simulation) wins,
                   -1 if leaf node player loses
        """
        _player = game_state.current_player

        # Play random moves until game ends
        while not game_state.game_over:
            legal_moves = game_state.get_legal_moves()
            if not legal_moves:
                break
            game_state.make_move(random.choice(legal_moves))

        # Determine winner from original player's perspective
        if game_state.winner == leaf_node_next_player:
            return -1.0
        else:
            return 1.0


    def _backpropagate(self, search_path, value):
        """
        Update all nodes in path from leaf to root

        IMPORTANT: Value flips at each level because players alternate!
        """
        # Start with the value from the leaf's perspective
        current_value = value

        for node in reversed(search_path):
            # Update node statistics
            node.visit_count += 1
            node.value_sum += current_value

            # Flip value for parent (opponent's perspective)
            current_value = -current_value

    def _get_move_probs(self, root, temperature=1.0):
        """Convert visit counts to move probabilities"""
        moves, visits = [], []

        # Only consider legal moves
        legal_moves = root.game_state.get_legal_moves()

        for move in legal_moves:
            if move in root.children:
                child = root.children[move]
                moves.append(move)
                visits.append(child.visit_count)
            else:
                # Moves that have never been visited
                moves.append(move)
                visits.append(0)

        # Apply temperature
        if temperature == 0:
            # Deterministic: choose move with most visits
            probs = np.zeros(len(moves))
            if visits:
                probs[np.argmax(visits)] = 1.0
        else:
            # Softmax with temperature
            visits = np.array(visits, dtype=np.float32)
            # Add small epsilon to avoid log(0)
            visits = visits + 1e-8
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits) if np.sum(visits) > 0 else np.ones(len(moves)) / len(moves)

        # Convert to board-sized array
        board_size = root.game_state.board_size
        action_probs = np.zeros(board_size * board_size)

        for move, prob in zip(moves, probs):
            move_idx = root.game_state.move_to_index(move)
            action_probs[move_idx] = prob

        return action_probs, root.value()

    def _log_state(self, root, iteration):
        """Log tree state - only root's children"""
        children_data = []

        # Get legal moves
        legal_moves = root.game_state.get_legal_moves()

        for move in legal_moves:
            if move in root.children:
                child = root.children[move]
                win_rate = child.value() if child.visit_count > 0 else 0.0

                # Calculate UCB1 for display
                if child.visit_count == 0:
                    ucb1 = 'inf'
                else:
                    q_value = child.value()
                    parent_visits = root.visit_count if root.visit_count > 0 else 1
                    u_value = (self.exploration_constant *
                              np.sqrt(np.log(parent_visits) / child.visit_count))
                    ucb1 = q_value + u_value
            else:
                # Not expanded yet
                win_rate = 0.0
                ucb1 = 'not expanded'

            children_data.append({
                'move': move,
                'visits': child.visit_count if move in root.children else 0,
                'value_sum': child.value_sum if move in root.children else 0,
                'win_rate': win_rate,
                'ucb1_score': ucb1,
                'depth': self._get_depth(child) if move in root.children else 0
            })

        children_data.sort(key=lambda x: x['move'])
        self.tree_log.append({
            'iteration': iteration,
            'root_visits': root.visit_count,
            'total_nodes': self._count_nodes(root),
            'max_depth': self._get_max_depth(root),
            'children': children_data
        })

    def _count_nodes(self, node):
        """Count total nodes in tree"""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node, current_depth=0):
        """Get maximum depth of tree"""
        if not node.children:
            return current_depth
        return max(self._get_max_depth(child, current_depth + 1)
                  for child in node.children.values())

    def _get_depth(self, node):
        """Get depth of this node"""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def _save_log(self, board_size):
        """Save log to JSON"""
        log_data = {
            'metadata': {
                'algorithm': 'NaiveMCTS',
                'num_simulations': self.num_simulations,
                'exploration_constant': self.exploration_constant,
                'board_size': board_size,
                'timestamp': datetime.now().isoformat()
            },
            'iterations': self.tree_log
        }
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Naive MCTS log saved to {self.log_file}")

# =============================================================================
# 5. ALPHAZERO MCTS
# =============================================================================

class AlphaZeroMCTS:
    """
    AlphaZero MCTS - Neural Network Guided Search
    ==============================================

    THE ULTIMATE ALGORITHM!

    Key Innovation: NO RANDOM SIMULATION!
    --------------------------------------
    Instead of random playouts, use neural network to:
    1. Predict which moves are good (policy)
    2. Evaluate position directly (value)

    Tree Structure (same as Naive MCTS):
    -------------------------------------
                         Root
                       /  |  \\
                      /   |   \\
                Child1  Child2  Child3
                   /\\      /\\
                  /  \\    /  \\
                 Grandchildren...

    Four Phases:
    ------------

    1. SELECTION (using PUCT, not UCB1)
       PUCT = UCB1 + Neural Network Prior

       PUCT(child) = Q(child) + c × P(child) × √(N_parent) / (1 + N_child)
                       ↑              ↑
                     Value         NN policy prior!

    2. EXPANSION
       Add children, each gets NN policy prior

    3. EVALUATION (NO simulation!)
       value = neural_network(position)  # Direct evaluation!

    4. BACKPROPAGATION
       Same as Naive MCTS

    Why It's Better:
    ----------------

    Naive MCTS:
    - 1 simulation = 50-200 random moves
    - Noisy result (random play is bad)
    - Needs many simulations to converge

    AlphaZero:
    - 1 "simulation" = 1 neural network forward pass (~1ms)
    - Learned evaluation (trained on millions of positions)
    - Much more accurate per simulation

    Result: 100 AlphaZero simulations ≈ 5000+ Naive MCTS simulations!

    Example Evolution:
    ------------------

    Iteration 1:
      NN says: Move A=0.4, B=0.3, C=0.2, D=0.1
      Tree visits: A=1, B=0, C=0, D=0

    Iteration 50:
      NN still says: A=0.4, B=0.3, C=0.2, D=0.1
      Tree visits: A=20, B=18, C=8, D=4
      → MCTS refined the policy! A and B are actually best

    Iteration 100:
      Final decision: Pick A (most visits)
      Training data: Position → MCTS visits [0.4, 0.36, 0.16, 0.08]
      → Train NN to match MCTS (not original NN output)!
    """

    def __init__(self, network, num_simulations=100, c_puct=1.0, device='cpu'):
        """
        Initialize AlphaZero MCTS

        Args:
            network: Neural network (policy + value)
            num_simulations: Number of MCTS iterations
            c_puct: Exploration constant (like UCB1's c)
            device: 'cpu' or 'cuda'
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, game_state):
        """
        Run AlphaZero MCTS search

        Returns:
            (move_probs, root_value): Visit-count distribution and position value
        """
        root = MCTSNode(game_state.clone())

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # ==================================================================
            # PHASE 1: SELECTION (using PUCT with NN priors)
            # ==================================================================
            while not node.is_leaf() and not node.game_state.game_over:
                move, node = self._select_child(node)
                search_path.append(node)

            # ==================================================================
            # PHASE 2 & 3: EXPANSION + EVALUATION (using NN, no simulation!)
            # ==================================================================
            value = self._expand_and_evaluate(node)

            # ==================================================================
            # PHASE 4: BACKPROPAGATION
            # ==================================================================
            self._backpropagate(search_path, value)

        # Return visit-count distribution
        move_probs = self._get_move_probs(root, temperature=1.0)
        return move_probs, root.value()

    def _select_child(self, node):
        """
        PUCT selection (UCB1 + neural network prior)

        PUCT Formula:
        -------------
        PUCT(child) = Q + c × P × √N_parent / (1 + N_child)

        Where P comes from neural network policy!
        """
        best_score = -float('inf')
        best_move, best_child = None, None

        for move, child in node.children.items():
            # Q value (exploitation)
            q_value = child.value()

            # U value (exploration, guided by NN prior)
            u_value = (self.c_puct * child.prior *
                      np.sqrt(node.visit_count) / (1 + child.visit_count))

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move, best_child = move, child

        return best_move, best_child

    def _expand_and_evaluate(self, node):
        """
        Expand node and evaluate using neural network

        NO RANDOM SIMULATION - This is the key innovation!

        Returns:
            value: NN evaluation of position
        """
        if node.game_state.game_over:
            # Terminal node
            #if node.game_state.winner == node.game_state.current_player:
            #    return 1.0
            #else:
            return -1.0

        # ==================================================================
        # Get neural network predictions
        # ==================================================================
        board_tensor = torch.FloatTensor(
            node.game_state.get_board_representation()
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_policy, value = self.network(board_tensor)
            policy = torch.exp(log_policy).cpu().numpy()[0]
            value = value.cpu().item()

        # ==================================================================
        # Expand node with NN policy priors
        # ==================================================================
        legal_moves = node.game_state.get_legal_moves()
        policy_sum = sum(policy[node.game_state.move_to_index(m)]
                        for m in legal_moves)

        for move in legal_moves:
            move_idx = node.game_state.move_to_index(move)
            prior = (policy[move_idx] / policy_sum if policy_sum > 0
                    else 1.0 / len(legal_moves))

            child_state = node.game_state.clone()
            child_state.make_move(move)

            child = MCTSNode(child_state, parent=node, move=move, prior=prior)
            node.children[move] = child

        node.is_expanded = True

        # Return NN value (no simulation needed!)
        return value

    def _backpropagate(self, search_path, value):
        """Update all nodes in path (same as Naive MCTS)"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def _get_move_probs(self, root, temperature=1.0):
        """
        Convert visit counts to probabilities

        Temperature:
        - 1.0: Proportional to visits
        - 0.0: Deterministic (argmax)
        - >1.0: More exploration
        """
        moves, visits = [], []
        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visit_count)

        if temperature == 0:
            probs = np.zeros(len(moves))
            probs[np.argmax(visits)] = 1.0
        else:
            visits = np.array(visits, dtype=np.float32)
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)

        action_probs = np.zeros(root.game_state.board_size ** 2)
        for move, prob in zip(moves, probs):
            action_probs[root.game_state.move_to_index(move)] = prob

        return action_probs

# =============================================================================
# 6. SELF-PLAY AND TRAINING
# =============================================================================

def play_game(network, board_size=5, num_simulations=100, device='cpu'):
    """
    Play one self-play game using AlphaZero MCTS

    Collects training data: (state, MCTS_policy, outcome)

    Returns:
        list: Training examples
    """
    game = HexGame(board_size)
    mcts = AlphaZeroMCTS(network, num_simulations=num_simulations, device=device)
    game_history = []

    while not game.game_over:
        # Get move from MCTS
        move_probs, _ = mcts.search(game)

        # Store training example
        game_history.append({
            'state': game.get_board_representation(),
            'policy': move_probs,
            'player': game.current_player
        })

        # Sample move
        legal_moves = game.get_legal_moves()
        legal_indices = [game.move_to_index(m) for m in legal_moves]
        legal_probs = move_probs[legal_indices]
        legal_probs = legal_probs / np.sum(legal_probs)

        move_idx = np.random.choice(len(legal_moves), p=legal_probs)
        move = legal_moves[move_idx]
        game.make_move(move)

    # Add outcome to all positions
    winner = game.winner
    for example in game_history:
        example['value'] = 1.0 if example['player'] == winner else -1.0

    return game_history

class AlphaZeroTrainer:
    """
    Training loop for AlphaZero

    The Virtuous Cycle:
    ===================

    1. Self-Play: Generate games using current NN + MCTS
       → Collects (state, MCTS_policy, outcome) tuples

    2. Training: Train NN on collected data
       → Policy loss: Match MCTS visit distribution
       → Value loss: Predict game outcome

    3. Improvement: Better NN → Better MCTS → Better self-play

    Repeat 100-1000 times!
    """

    def __init__(self, board_size=5, num_res_blocks=5, device='cpu'):
        self.board_size = board_size
        self.device = device
        self.network = HexNetwork(board_size, num_res_blocks).to(device)
        self.optimizer = Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        self.replay_buffer = deque(maxlen=10000)

    def train(self, num_iterations=100, games_per_iteration=10,
              num_simulations=100, batch_size=32, epochs_per_iteration=10):
        """Main training loop"""
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # 1. Self-play
            print("Generating self-play games...")
            iteration_data = []
            for game_num in range(games_per_iteration):
                game_data = play_game(self.network, self.board_size,
                                     num_simulations, self.device)
                iteration_data.extend(game_data)

            self.replay_buffer.extend(iteration_data)
            print(f"Replay buffer size: {len(self.replay_buffer)}")

            # 2. Train
            print("Training network...")
            avg_policy_loss, avg_value_loss = self._train_network(
                batch_size, epochs_per_iteration)
            print(f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

            # 3. Save checkpoint
            if (iteration + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_iter_{iteration + 1}.pth")

    def _train_network(self, batch_size=32, epochs=10):
        """Train network on replay buffer"""
        self.network.train()
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for epoch in range(epochs):
            batch = (list(self.replay_buffer) if len(self.replay_buffer) < batch_size
                    else random.sample(self.replay_buffer, batch_size))

            states = torch.FloatTensor(np.array([ex['state'] for ex in batch])).to(self.device)
            target_policies = torch.FloatTensor([ex['policy'] for ex in batch]).to(self.device)
            target_values = torch.FloatTensor([[ex['value']] for ex in batch]).to(self.device)

            log_policies, pred_values = self.network(states)

            policy_loss = -torch.sum(target_policies * log_policies) / batch_size
            value_loss = F.mse_loss(pred_values, target_values)
            total_loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        return total_policy_loss / num_batches, total_value_loss / num_batches

    def _save_checkpoint(self, filename):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint: {filename}")

# =============================================================================
# 7. PLAYER CLASSES
# =============================================================================

class HexPlayer:
    """Base player class"""
    def get_move(self, game_state):
        raise NotImplementedError

class PureMonteCarloPlayer(HexPlayer):
    """Player using Pure Monte Carlo"""
    def __init__(self, num_simulations=1000):
        self.monte_carlo = PureMonteCarlo(num_simulations)

    def get_move(self, game_state):
        return self.monte_carlo.get_move(game_state)

class NaiveMCTSPlayer(HexPlayer):
    """Player using Naive MCTS (deep tree, random rollouts)"""
    def __init__(self, num_simulations=1000, exploration_constant=1.4,
                 log_tree=False, log_file=None):
        self.mcts = NaiveMCTS(num_simulations, exploration_constant,
                             log_tree, log_file)

    def get_move(self, game_state):
        move_probs, _ = self.mcts.search(game_state)
        legal_moves = game_state.get_legal_moves()
        legal_indices = [game_state.move_to_index(m) for m in legal_moves]
        best_idx = legal_indices[np.argmax(move_probs[legal_indices])]
        return game_state.index_to_move(best_idx)

class AlphaZeroPlayer(HexPlayer):
    """Player using AlphaZero (deep tree, NN guidance)"""
    def __init__(self, network, num_simulations=100, device='cpu'):
        self.mcts = AlphaZeroMCTS(network, num_simulations, device=device)

    def get_move(self, game_state):
        move_probs, _ = self.mcts.search(game_state)
        legal_moves = game_state.get_legal_moves()
        legal_indices = [game_state.move_to_index(m) for m in legal_moves]
        best_idx = legal_indices[np.argmax(move_probs[legal_indices])]
        return game_state.index_to_move(best_idx)

class HumanPlayer(HexPlayer):
    """Human player"""
    def get_move(self, game_state):
        legal_moves = game_state.get_legal_moves()
        print("Legal moves:", legal_moves)
        while True:
            try:
                move_str = input("Enter move as 'row,col': ")
                row, col = map(int, move_str.split(','))
                move = (row, col)
                if move in legal_moves:
                    return move
                print("Invalid move.")
            except:
                print("Invalid format. Use 'row,col'.")

# =============================================================================
# 8. TOURNAMENT AND COMPARISON
# =============================================================================

def play_tournament(player1, player2, num_games=10, board_size=5):
    """
    Play tournament between two players

    Returns:
        (player1_wins, player2_wins)
    """
    player1_wins = 0
    player2_wins = 0

    # Get player names
    def get_name(player):
        if isinstance(player, PureMonteCarloPlayer):
            return "PureMC"
        elif isinstance(player, NaiveMCTSPlayer):
            return "NaiveMCTS"
        elif isinstance(player, AlphaZeroPlayer):
            return "AlphaZero"
        elif isinstance(player, HumanPlayer):
            return "Human"
        return "Unknown"

    p1_name = get_name(player1)
    p2_name = get_name(player2)

    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        game = HexGame(board_size)

        # Alternate starting player
        if game_num % 2 == 0:
            current, other = player1, player2
            curr_name, other_name = p1_name, p2_name
        else:
            current, other = player2, player1
            curr_name, other_name = p2_name, p1_name

        while not game.game_over:
            if game.current_player == PLAYER_BLUE:
                move = current.get_move(game)
                print(f"{curr_name} (Blue) plays: {move}")
            else:
                move = other.get_move(game)
                print(f"{other_name} (Red) plays: {move}")
            game.make_move(move)

        # Determine winner
        if (game.winner == PLAYER_BLUE and game_num % 2 == 0) or \
           (game.winner == PLAYER_RED and game_num % 2 == 1):
            player1_wins += 1
            print(f"{p1_name} wins!")
        else:
            player2_wins += 1
            print(f"{p2_name} wins!")

    print(f"\n=== Tournament Results ===")
    print(f"{p1_name}: {player1_wins}")
    print(f"{p2_name}: {player2_wins}")

    return player1_wins, player2_wins

# =============================================================================
# 9. TREE VISUALIZATION
# =============================================================================

def load_tree_log(filename):
    """Load tree log from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)

def visualize_tree_iteration(log_data, iteration_num):
    """
    Print tree state at specific iteration

    Shows: move, visits, wins, win rate, UCB1, selection
    """
    iteration = log_data['iterations'][iteration_num]

    print(f"\n{'='*70}")
    print(f"Iteration {iteration['iteration']}")
    print(f"Root visits: {iteration['root_visits']}")
    if 'total_nodes' in iteration:
        print(f"Total nodes: {iteration['total_nodes']}")
        print(f"Max depth: {iteration['max_depth']}")
    print(f"{'='*70}\n")

    print(f"{'Move':<10} {'Visits':<8} {'Value':<10} {'WinRate':<10} {'UCB1':<10} {'Sel':<5}")
    print("-" * 70)

    for child in iteration['children']:
        sel = "✓" if child.get('selected', False) else ""
        ucb1 = (f"{child['ucb1_score']:.4f}" if isinstance(child['ucb1_score'], (int, float))
                else "∞")
        depth_str = f" (d={child['depth']})" if 'depth' in child else ""

        print(f"{str(child['move']):<10} "
              f"{child['visits']:<8} "
              f"{child['value_sum']:<10.1f} "
              f"{child['win_rate']:<10.3f} "
              f"{ucb1:<10} "
              f"{sel}{depth_str}")

def compare_algorithms_visually(log_files, iteration_num):
    """
    Compare multiple algorithms side-by-side at same iteration

    Args:
        log_files: Dict like {'NaiveMCTS': 'naive.json'}
        iteration_num: Which iteration to compare
    """
    fig, axes = plt.subplots(1, len(log_files), figsize=(6*len(log_files), 5))
    if len(log_files) == 1:
        axes = [axes]

    for idx, (name, filename) in enumerate(log_files.items()):
        log_data = load_tree_log(filename)
        iteration = log_data['iterations'][iteration_num]
        ax = axes[idx]

        moves = [str(child['move']) for child in iteration['children']]
        visits = [child['visits'] for child in iteration['children']]
        win_rates = [child['win_rate'] for child in iteration['children']]

        x = np.arange(len(moves))
        width = 0.35

        ax.bar(x - width/2, visits, width, label='Visits', alpha=0.8)
        ax.bar(x + width/2, [wr * max(visits) if visits else 0 for wr in win_rates],
               width, label='Win Rate (scaled)', alpha=0.8)

        ax.set_xlabel('Move')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\nIteration {iteration["iteration"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(moves, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def animate_tree_growth(log_data, save_as=None):
    """
    Create animation showing tree growth over iterations

    Args:
        log_data: Loaded tree log
        save_as: Optional filename to save (e.g., 'growth.gif')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def update(frame):
        ax1.clear()
        ax2.clear()

        iteration = log_data['iterations'][frame]

        # Plot 1: Visit counts
        moves = [str(child['move']) for child in iteration['children']]
        visits = [child['visits'] for child in iteration['children']]

        ax1.bar(moves, visits)
        ax1.set_xlabel('Move')
        ax1.set_ylabel('Visit Count')
        ax1.set_title(f'Visit Distribution (Iteration {iteration["iteration"]})')
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Win rates
        win_rates = [child['win_rate'] for child in iteration['children']]
        colors = ['green' if wr > 0.5 else 'red' if wr < 0.5 else 'gray'
                 for wr in win_rates]

        ax2.bar(moves, win_rates, color=colors, alpha=0.7)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Move')
        ax2.set_ylabel('Win Rate')
        ax2.set_title(f'Win Rate by Move')
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

    anim = animation.FuncAnimation(fig, update, frames=len(log_data['iterations']),
                                   interval=200, repeat=True)

    if save_as:
        anim.save(save_as, writer='pillow', fps=5)
        print(f"Animation saved to {save_as}")

    plt.show()

# =============================================================================
# 10. EXAMPLE USAGE
# =============================================================================

def example_1():
    print("="*70)
    print("HEX ALPHAZERO - EXAMPLE USAGE")
    print("="*70)

    # Example 1: Compare Pure MC and Naive MCTS
    print("\n1. Creating players...")
    pure_mc = PureMonteCarloPlayer(num_simulations=500)
    naive_mcts = NaiveMCTSPlayer(num_simulations=4000, exploration_constant=1.4,
                                 log_tree=True, log_file='naive_mcts_log.json')

    print("\n2. Running Pure MC vs Naive MCTS tournament...")
    play_tournament(pure_mc, naive_mcts, num_games=1, board_size=5)

    print("\n3. Visualizing tree logs...")
    # Visualize Pure MC
    pure_log = load_tree_log('pure_mc_log.json')
    print("\nPure MC - Iteration 0:")
    visualize_tree_iteration(pure_log, 0)
    print("\nPure MC - Final iteration:")
    visualize_tree_iteration(pure_log, -1)

    # Visualize Naive MCTS
    naive_log = load_tree_log('naive_mcts_log.json')
    print("\nNaive MCTS - Iteration 0:")
    visualize_tree_iteration(naive_log, 0)
    print("\nNaive MCTS - Final iteration:")
    visualize_tree_iteration(naive_log, -1)

    # Compare side-by-side
    compare_algorithms_visually({
        'PureMC': 'pure_mc_log.json',
        'NaiveMCTS': 'naive_mcts_log.json'
    }, iteration_num=-1)

    # Create animations
    print("\n4. Creating animations...")
    animate_tree_growth(pure_log, save_as='pure_mc_growth.gif')
    animate_tree_growth(naive_log, save_as='naive_mcts_growth.gif')

    print("\nDone! Check the generated .gif files and .json logs.")


"""
================================================================================
VISUAL GAME EXAMPLE - Watch Two Algorithms Play Against Each Other
================================================================================

This script shows two algorithms playing a complete game with:
- Visual board display after each move
- Move-by-move commentary
- Final game analysis
- Optional animation

================================================================================
"""

# =============================================================================
# VISUAL GAME PLAYER
# =============================================================================

class VisualGameRunner:
    """
    Play a visual game between two algorithms

    Shows the board after each move with:
    - Colored hexagons for each player
    - Move numbers
    - Current player indicator
    - Win detection
    """

    def __init__(self, board_size=5):
        self.board_size = board_size
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.ion()  # Interactive mode

    def display_board(self, game, move_number, p1_name, p2_name):
        """
        Display current board state

        Args:
            game: HexGame instance
            move_number: Current move count
            p1_name: Player 1 name
            p2_name: Player 2 name
        """
        self.ax.clear()

        # Draw hexagons
        for i in range(self.board_size):
            for j in range(self.board_size):
                x = j + HEX_HORIZONTAL_OFFSET * i
                y = i * HEX_VERTICAL_SPACING

                hex_vertices = HEX_VERTICES + [x, y]
                hexagon = patches.Polygon(hex_vertices, fill=True,
                                        edgecolor='black', linewidth=2)

                # Color based on occupancy
                if game.board[i, j] == PLAYER_BLUE:
                    hexagon.set_facecolor('dodgerblue')
                    hexagon.set_edgecolor('darkblue')
                elif game.board[i, j] == PLAYER_RED:
                    hexagon.set_facecolor('crimson')
                    hexagon.set_edgecolor('darkred')
                else:
                    hexagon.set_facecolor('lightgray')
                    hexagon.set_edgecolor('gray')

                self.ax.add_patch(hexagon)

                # Add cell labels
                if game.board[i, j] == 0:
                    self.ax.text(x, y, f'{i},{j}', ha='center', va='center',
                               fontsize=7, alpha=0.5, color='black')

        # Add border indicators
        max_x = self.board_size + HEX_HORIZONTAL_OFFSET * (self.board_size - 1)
        max_y = (self.board_size - 1) * HEX_VERTICAL_SPACING

        # Blue borders (left-right)
        self.ax.plot([-0.7, -0.7], [-0.5, max_y + 0.5],
                    color='blue', linewidth=4, label='Blue: Left↔Right')
        self.ax.plot([max_x + 0.7, max_x + 0.7], [-0.5, max_y + 0.5],
                    color='blue', linewidth=4)

        # Red borders (top-bottom)
        self.ax.plot([-0.5, max_x + 0.5], [-0.7, -0.7],
                    color='red', linewidth=4, label='Red: Top↔Bottom')
        self.ax.plot([-0.5, max_x + 0.5], [max_y + 0.7, max_y + 0.7],
                    color='red', linewidth=4)

        # Set limits
        self.ax.set_xlim(-1, max_x + 1)
        self.ax.set_ylim(-1.2, max_y + 1.2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Title with game info
        if game.game_over:
            winner_name = p1_name if game.winner == PLAYER_BLUE else p2_name
            winner_color = "BLUE" if game.winner == PLAYER_BLUE else "RED"
            title = f"🏆 GAME OVER - {winner_name} ({winner_color}) WINS! 🏆\n"
            title += f"Total moves: {move_number}"
            color = 'blue' if game.winner == PLAYER_BLUE else 'red'
        else:
            current = p1_name if game.current_player == PLAYER_BLUE else p2_name
            color_name = "BLUE" if game.current_player == PLAYER_BLUE else "RED"
            title = f"Move {move_number + 1}: {current}'s turn ({color_name})\n"
            title += f"{p1_name} (Blue) vs {p2_name} (Red)"
            color = 'black'

        self.ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=20)

        plt.draw()
        plt.pause(0.5)  # Pause to see the move

def play_visual_game(player1, player2, board_size=5, pause_time=1.0):
    """
    Play a complete game with visual display

    Args:
        player1: First player (plays Blue - left/right)
        player2: Second player (plays Red - top/bottom)
        board_size: Board size
        pause_time: Seconds to pause between moves

    Returns:
        game: Completed HexGame instance
    """
    # Get player names
    def get_name(player):
        if isinstance(player, PureMonteCarloPlayer):
            return "PureMC"
        elif isinstance(player, NaiveMCTSPlayer):
            return "NaiveMCTS"
        elif isinstance(player, AlphaZeroPlayer):
            return "AlphaZero"
        elif isinstance(player, HumanPlayer):
            return "Human"
        return "Unknown"

    p1_name = get_name(player1)
    p2_name = get_name(player2)

    print("\n" + "="*70)
    print(f"STARTING VISUAL GAME: {p1_name} (Blue) vs {p2_name} (Red)")
    print("="*70)
    print(f"Board size: {board_size}×{board_size}")
    print(f"Blue wins by connecting LEFT ↔ RIGHT")
    print(f"Red wins by connecting TOP ↔ BOTTOM")
    print("="*70 + "\n")

    game = HexGame(board_size)
    visualizer = VisualGameRunner(board_size)

    # Display initial empty board
    visualizer.display_board(game, 0, p1_name, p2_name)
    time.sleep(pause_time)

    move_count = 0

    while not game.game_over:
        # Determine current player
        if game.current_player == PLAYER_BLUE:
            current_player = player1
            current_name = p1_name
            color_name = "Blue"
        else:
            current_player = player2
            current_name = p2_name
            color_name = "Red"

        # Get move
        print(f"Move {move_count + 1}: {current_name} ({color_name}) thinking...")
        move = current_player.get_move(game)
        print(f"  → {current_name} plays at {move}")

        # Make move
        game.make_move(move)
        move_count += 1

        # Display board
        visualizer.display_board(game, move_count, p1_name, p2_name)

        # Check for win
        if game.game_over:
            winner_name = p1_name if game.winner == PLAYER_BLUE else p2_name
            winner_color = "Blue" if game.winner == PLAYER_BLUE else "Red"
            print(f"\n{'='*70}")
            print(f"🏆 GAME OVER! {winner_name} ({winner_color}) WINS! 🏆")
            print(f"{'='*70}")
            print(f"Total moves: {move_count}")
            print(f"Winner: {winner_name}")
            print(f"Connection: {winner_color} connected their sides!")
            print("="*70 + "\n")

        time.sleep(pause_time)

    # Keep final board displayed
    plt.ioff()
    plt.show()

    return game


# =============================================================================
# GAME ANALYSIS
# =============================================================================

def analyze_game(game, p1_name, p2_name):
    """
    Analyze completed game

    Args:
        game: Completed HexGame
        p1_name: Player 1 name
        p2_name: Player 2 name
    """
    print("\n" + "="*70)
    print("GAME ANALYSIS")
    print("="*70)

    # Count stones
    blue_stones = np.sum(game.board == PLAYER_BLUE)
    red_stones = np.sum(game.board == PLAYER_RED)
    empty_cells = np.sum(game.board == 0)

    print(f"\nStone count:")
    print(f"  {p1_name} (Blue): {blue_stones} stones")
    print(f"  {p2_name} (Red):  {red_stones} stones")
    print(f"  Empty cells:      {empty_cells}")
    print(f"  Total moves:      {blue_stones + red_stones}")

    # Determine winner
    winner_name = p1_name if game.winner == PLAYER_BLUE else p2_name
    winner_color = "Blue" if game.winner == PLAYER_BLUE else "Red"

    print(f"\nResult:")
    print(f"  Winner: {winner_name} ({winner_color})")
    print(f"  Connection: {winner_color} connected their opposite sides")

    # Move efficiency
    total_cells = game.board_size ** 2
    fill_percentage = ((blue_stones + red_stones) / total_cells) * 100

    print(f"\nBoard fill: {fill_percentage:.1f}% ({blue_stones + red_stones}/{total_cells} cells)")

    if fill_percentage < 50:
        print("  → Quick decisive game!")
    elif fill_percentage < 75:
        print("  → Moderate length game")
    else:
        print("  → Long tactical battle!")

    print("="*70)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_visual_games(board_size=7):
    """
    Example: Watch different algorithm matchups
    """

    print("\n" + "="*70)
    print("VISUAL GAME EXAMPLES")
    print("="*70)

    # Example 1: Naive MCTS vs Pure MCTS
    print("\n\n>>> EXAMPLE 3: Naive MCTS vs Pure MC")
    print("-"*70)

    player1 = NaiveMCTSPlayer(num_simulations=2000, exploration_constant=1.4)
    player2 = PureMonteCarloPlayer(num_simulations=500)

    game1 = play_visual_game(player1, player2, board_size=board_size, pause_time=0.8)
    analyze_game(game1, "NaiveMCTS", "PureMC")

    print("\n\n>>> ALL EXAMPLES COMPLETED!")
    print("="*70)

# =============================================================================
# QUICK MATCH FUNCTION
# =============================================================================

def quick_match(algo1_name, algo2_name, board_size=7, simulations=4000):
    """
    Quick visual match between two algorithms

    Args:
        algo1_name: 'PureMC', 'NaiveMCTS', or 'AlphaZero'
        algo2_name: Same options
        board_size: Board size
        simulations: Simulations per move

    Example:
        quick_match('PureMC', 'NaiveMCTS', board_size=7, simulations)
    """
    # Create players
    players = {
        'PureMC': lambda: PureMonteCarloPlayer(num_simulations=simulations//10),
        'NaiveMCTS': lambda: NaiveMCTSPlayer(num_simulations=simulations*2,
                                             exploration_constant=1.414),
        #'AlphaZero': lambda: AlphaZeroPlayer(network, num_simulations=simulations),
    }

    if algo1_name not in players or algo2_name not in players:
        print(f"Error: Choose from {list(players.keys())}")
        return

    print(f"\n🎮 QUICK MATCH: {algo1_name} vs {algo2_name}")
    print(f"Board: {board_size}×{board_size}, Simulations: {simulations}")

    player1 = players[algo1_name]()
    player2 = players[algo2_name]()

    game = play_visual_game(player1, player2, board_size=board_size, pause_time=0.3)
    analyze_game(game, algo1_name, algo2_name)

    return game

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         VISUAL HEX GAME - WATCH ALGORITHMS COMPETE                ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Available examples:

    1. example_visual_games()
       → Watch all algorithm combinations

    2. quick_match('PureMC', 'NaiveMCTS', board_size=7, simulations=4000)
       → Quick single game

    3. Custom game:
       player1 = PureMonteCarloPlayer(500)
       player2 = NaiveMCTSPlayer(2000)
       game = play_visual_game(player1, player2, board_size=7)
       analyze_game(game, "PureMC", "NaiveMCTS")
    """)

    # Run example
    choice = input("Run a quick match? (y/n): ").strip().lower()
    if choice == 'y':
        quick_match('PureMC', 'NaiveMCTS', board_size=7)
    else:
        example_visual_games()
