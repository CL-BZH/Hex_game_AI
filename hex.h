#ifndef HEX_H
#define HEX_H

// Not yet implemented: use of ncurses
//#ifdef _NCURSES
//#include <curses.h>
//#endif

#include <sstream> 
#include <string>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <random>

#include <cctype>

#include "graph.h"
#include "dfs.h"
#include "shortestpath.h"

// Maximun number of palyers
const unsigned int max_players{2};

// Players IDs
const unsigned int blue_player{0};
const unsigned int red_player{1};

// Four type of cells on the board
enum class NodeValue: unsigned int {
  B = blue_player, // Defines a node already selected by the Blue player
  R = red_player,  // Defines a node already selected by the Red player
  AVAILABLE,       // Defines a node that is free
  OUTSIDE,         // Defines a node that is ouside the board
};

struct PlayersColors {
  // colors[blue_player] = "Blue", colors[red_player] = "Red"
  static const std::string colors[max_players];
  
  static const std::string color(unsigned int player) {
    return colors[player];
  }
  
  // Accessor
  char operator[](NodeValue node_value) const {
    return (*this)[static_cast<int>(node_value)];
  }
  char operator[](unsigned int player_id) const {
    if(!((player_id == blue_player) ||  (player_id == red_player))) {
      throw std::runtime_error{"Invalid call to PlayerColors[]"};
    }
    return colors[player_id][0];
  }
};

const std::string PlayersColors::colors[]{"Blue", "Red"};

enum class PlayerType {
  Human,
  Machine
};

struct Hex {
  
  // Build the Hex board with 'size' hexagonal cells. Each cell is represented
  // by a node in a graph. Hence the graph has 'size'*'size' nodes.
  // But I add surrounding cells around the board (see explanations below),
  // so the graph will have (size + 2)^2 nodes.
  // By default the board is of size 11 by 11 and 2 human players play
  // against each other.
  Hex(unsigned int size=11, unsigned int nb_players=max_players):
    board_size{size}, nb_human_players{nb_players},
    nb_machine_players{max_players - nb_human_players},
    graph_size{(size+2)*(size+2)},
    graph{graph_size, EdgeColor::Any},
    gen(rd()),
    uniform_distribution(std::uniform_int_distribution<int>(0, board_size-1)) {
      
      if (size < 3)
	throw std::runtime_error{"The board's size should be at least 3"};

      if (!((nb_players == 0) || (nb_players == 1) || (nb_players == 2)))
	throw std::runtime_error{"The number of players can be 0, 1 or 2"};

      if(nb_machine_players == max_players) {
	// The machine plays against itself
	machine_players.push_back(blue_player);
	machine_players.push_back(red_player);
      }

      // Initialize the nodes
      init_nodes();
  
      // Build the graph that represent the Hex game (i.e. add edges)
      build_graph();

      // Draw the empty board
      draw_board();
      
      // Start the game
      start();
      
    }

  /* 
   * Draw the board using ascii symbols
   * e.g. for a 3x3 board:
   *  . - . - .
   *   \ / \ / \
   *    . - . - .
   *     \ / \ / \
   *      . - . - .
   *
   * The positions of the Blue and Red players are indicated by 'B'
   * and 'R' character respectively.
   */
  void draw_board(const std::vector<Node>* route=nullptr) {

    std::stringstream ss;
    
    for(unsigned int row{1}; row < board_size+1; ++row) {
      // Draw the nodes' content ('.', 'R' or 'B') and West-East edges
      ss << std::string(2*row+1, ' ');
      for(unsigned int column{1}; column < board_size; ++column) {
	unsigned int node_id{row*(board_size+2) + column};
	char node_char{get_node_char(row, column)};
	if (route != nullptr) {
	  bool found{false};
	  for (auto node: *route) {
	    if(node.id == node_id) {
	      found = true;
	      break;
	    }
	  }
	  if(found) {
	    // The node is on the route => change the character to lowercase
	    node_char = tolower(node_char);
	  }
	}
	ss << node_char;
	ss << ' ';
	ss << '-';
	ss << ' ';
      }

      unsigned int node_id{row*(board_size+2) + board_size};
      char node_char{get_node_char(row, board_size)};
      if (route != nullptr) {
	bool found{false};
	for (auto node: *route) {
	  if(node.id == node_id) {
	      found = true;
	      break;
	  }
	}
	if(found) 
	  node_char = tolower(node_char);
      }
      ss << node_char;
      ss << std::endl;
      if(row == board_size)
	break;
      // Draw the North-South edges
      ss << std::string(2*row+2, ' ');
      for(unsigned int column{1}; column < board_size; ++column) {
	ss << '\\';
	ss << " ";
	ss << '/';
	ss << ' ';
      }
      ss << '\\';
      ss << std::endl;
    }
    std::cout << ss.str();
  }

private:
  
  // The board size (number of cell per edge. 11 by default)
  unsigned int board_size{11};

  // The graph size
  unsigned int graph_size;

  // The number of human players
  // 0: the machine plays against itself
  // 1: the machine is the other player
  // 2: 2 players play against each other
  unsigned int nb_human_players;

  // The number of machine players (2 - nb_human_players)
  unsigned int nb_machine_players;
  
  // Store the id of the machine player(s)
  std::vector<unsigned int> machine_players;

  // number of selected cells
  unsigned int nb_selected_cells{0};

  // obtain a random number from hardware
  std::random_device rd;
  
  // generator
  std::mt19937 gen;
 
  // Uniform distibution over the range [0, board_size - 1]
  // Note: std::uniform_int_distribution<> is inclusive
  std::uniform_int_distribution<int> uniform_distribution;
  
  enum class EdgeColor {
    Blue,
    Red,
    Any,
  };
  
  // The graph that represent the game
  // The graph object represent the topology of the graph (edges between nodes).
  // No node is actually stored in the graph. (see Graph class in graph.h)
  ColoredGraph<EdgeColor> graph;

  // Store the nodes. I use a trick to make life easier for the addition of the
  // nodes' edges. I add one layer of cells around the board. The nodes
  // representing these cells are marked as OUTSIDE.
  // For example a 3x3 boards will be represented with surrounded cells like this:
  // O O O O O
  //  O I I I O
  //   O I I I O
  //    O I I I O
  //     O O O O O
  // With 'O' the ouside nodes and I the nodes on the board.
  std::vector<Node> nodes;

  // Select a position on the board.
  // Returns 'true' if a valid position was selected (false otherwise)
  bool select(unsigned int board_row,
	      unsigned int board_column,
	      unsigned int player_id,
	      PlayerType player_type) {

    if (!((player_id == blue_player) || (player_id == red_player)))
      throw std::runtime_error{"Invalid player's Id"};

    if(!((player_type == PlayerType::Human) ||
	 (player_type == PlayerType::Machine)))
      throw std::runtime_error{"Invalid player's type"};
      
    std::stringstream err;
    
    // Convert board's row and column to node row and column
    // (that is because I added a layer of 'OUTSIDE' nodes around the board)
    unsigned int node_row{board_row+1};
    unsigned int node_column{board_column+1};
    unsigned int node_id{node_row*(board_size+2) + node_column};
    NodeValue node_value;
    
    if((board_row >= board_size) || (board_column >= board_size) ||
       (!is_on_board(node_id))) {
      err << "Not valid row or column";
      goto Error;
    }

    node_value= static_cast<NodeValue>(nodes[node_id].value);
    
    if (node_value != NodeValue::AVAILABLE) {
      err << "Already selected position";
      goto Error;
    }
    
    // Register the player has owner of this selected cell
    nodes[node_id].value = static_cast<double>(player_id);
    
    // Increase by 1 the total number of selected cells
    ++nb_selected_cells;
    
    return true;

  Error:
    
    if(player_type == PlayerType::Human)
      std::cerr << err.str() << std::endl;

    return false;
  }

  // Initialization of the graph's node
  // All nodes representing cells on the board are marked AVAILABLE.
  // All nodes representing the cells surrounding the board are marked OUTSIDE
  void init_nodes() {
    unsigned int total_size{graph_size + 4*board_size + 4};
    nodes.resize(total_size);
    for(unsigned int row{0}; row < board_size+2; ++row) {
       for(unsigned int column{0}; column < board_size+2; ++column) {
	 unsigned int node_id{row*(board_size+2) + column};
	 nodes[node_id].id = node_id;
	 // Mark the nodes for the cells surrounding the board as outside
	 if((row == 0) || (row == board_size+1) ||
	    (column == 0) || (column == board_size+1))
	   nodes[node_id].value = static_cast<double>(NodeValue::OUTSIDE);
	 else
	   nodes[node_id].value = static_cast<double>(NodeValue::AVAILABLE);
       }
    }
  }

  // Tell if a node is on the board or surrounding it
  bool is_on_board(unsigned int node_id) {
    return nodes[node_id].value != static_cast<double>(NodeValue::OUTSIDE);
  }
  bool is_on_board(unsigned int node_row, unsigned int node_column) {
    unsigned int node_id{node_row*(board_size+2) + node_column};
    return is_on_board(node_id);
  }

  // Tells if the board is full
  bool board_is_full() {
    return (nb_selected_cells == graph_size);
  }
  
  // Build the graph representing the board, such that each internal hexagon
  // (a node) has six neighbors (so each node would have 6 edges). And the top
  // left corner hexagon and the bottom right corner hexagon have two neighbors,
  // the top right corner hexagon and the bottom left corner hexagon have three
  // neighbors and a non-corner board's edge has 4 neighbors.
  void build_graph() {
    // A position on the board is represented by a unique pair (row, column)
    // which gives a node id in the graph. row and column for valid nodes (i.e
    // the nodes that are on the board) are in [1, board_size]
    for(unsigned int row{1}; row < board_size+1; ++row) {
       for(unsigned int column{1}; column < board_size+1; ++column) {
	 unsigned int node_id{row*(board_size+2) + column};
	 unsigned int neighbor_id;
	 unsigned int neighbor_row;
	 unsigned int neighbor_col;
	 for(int row_offset{0}; row_offset < 2; ++row_offset) {
	   for(int col_offset{-1}; col_offset < 2; ++col_offset) {
	     // Get the row and column of the surrounding nodes
	     if(row_offset == col_offset)
	       continue;
	     neighbor_row = row + row_offset;
	     neighbor_col = column + col_offset;
	     neighbor_id = neighbor_row*(board_size+2) + neighbor_col;
	     if(neighbor_id == node_id - 1)
	       continue;//This edge was already added
	     if(is_on_board(neighbor_id)) {
	       // The neighbor node is on the board so we can add an edge
	       graph.add_edge(nodes[node_id], nodes[neighbor_id], 1);
	       //std::cout << "Add edge between node " << node_id;
	       //std::cout << " and node " << neighbor_id << std::endl;
	     }
	   }
	 }
      }
    }    
  }

  // Helper function to represent the board's cells
  char get_node_char(unsigned int row, unsigned int column) {
    unsigned int node_id{row*(board_size+2) + column};
    NodeValue node_value{static_cast<NodeValue>(nodes[node_id].value)};
    char ret;
    PlayersColors players_colors;
    switch(node_value) {
    case NodeValue::AVAILABLE:
      ret = '.';
      break;
    case NodeValue::B:
      ret = players_colors[node_value];
      break;
    case NodeValue::R:
      ret = players_colors[node_value];
      break;
    default:
      throw std::runtime_error{"Unknown node value"};
    }
    return ret;
  }

  void set_node_value(unsigned int node_id, NodeValue value) {
    nodes[node_id].value = static_cast<double>(value);
  }

  // When there is 1 or 2 players, 1 player has to choose its color.
  void color_selection() {
    unsigned int player;
    unsigned int oponent;

    // Invite one of the player to select its color
    std::cout << "Select your color (Blue starts)\n";
    std::cout << PlayersColors::color(blue_player) << " enter "
	      << blue_player << ", ";
    std::cout << PlayersColors::color(red_player) << " enter "
	      << red_player << "\n";
    
    while(true) {
      std::cin >> player;
      if((player == blue_player) || (player == red_player))
	break;
      std::cout << "Please enter " << blue_player << " or " << red_player << std::endl;
    }
    
    if(player == blue_player) {
      oponent = red_player;
      //p1 selected Blue => p1 starts
      std::cout << "You selected the color Blue. You start.\n"
		<< "To win you must create a path between West and East.\n"
		<< "Your oponent must build a path between North and South.\n\n";
    } else {
      oponent = blue_player;
      std::cout << "You selected the color Red. Your oponent will start.\n"
		<< "To win you must create a path between North and South.\n"
		<< "Your oponent must build a path between West and East.\n\n";
    }

    // If there is 1 machine player store the id of the machine player
    if(nb_machine_players == 1)
      machine_players.push_back(oponent);
  }

  // Invite the player to enter the position he wants to select on the board
  void prompt_player(unsigned int human_player_id,
		     unsigned int& row, unsigned int& col) {
    
    // Give an example of how to enter a position if it is the first time
    static bool give_example{true};
    
    // Check if a valid position was selected
    bool position_is_selected{false};

    // Loop until the player selects a valid position on the board
    while(!position_is_selected) {
      std::cout << PlayersColors::color(human_player_id)
		<< " player selects a position ";
      
      if(give_example) {
	std::cout << "(Example for row 0 column 1 enter: 0 1):\n";
	give_example = false;
      }
      
      std::cin.ignore();
      std::cin >> row >> col;
      // Loop until valid row and column are entered
      if (select(row, col, human_player_id, PlayerType::Human))
	break;
    }
  }

  // The machine plays against itself or again a human player
  void machine_play(unsigned int machine_player_id,
		    unsigned int& row, unsigned int& col) {
    // Check if a valid position was selected
    bool position_is_selected{false};
    
    // Loop until the machine selects a valid position on the board
    while(!position_is_selected) {
      machine_engine(row, col);
      if (select(row, col, machine_player_id, PlayerType::Machine))
	break;
    }
  }

  // Engine for the machine to select a row and a column on the board
  void machine_engine(unsigned int& row, unsigned int& col) {
    // Stupid machine:
    // generate random numbers for the row and the column
    row = uniform_distribution(gen); 
    col = uniform_distribution(gen); 
  }

  // Start the game
  void start() {
    
    // Ask one player to select its color if the machine does
    // not play against itself (the oponent will be given the other color).
    if(nb_machine_players != max_players)
      color_selection();

    // The Blue player always starts
    unsigned int player{blue_player};
    
    unsigned int row, col;
    bool game_end{false};

    // Run the game until there is a winner or the board is full
    while(!game_end) {
      
      // Invite human player to play if the machine does not play alone.
      // There are 2 cases:
      //     2 humans play against each other
      //     or
      //     1 human plays against the machine and it is his turn
      //     to select a position on the board
      if((nb_human_players == max_players) ||
	 ((nb_human_players == 1) && (player != machine_players[0]))) {
	//Invite the player to select a position 
	prompt_player(player, row, col);
      } else {
	// Either the machine plays against itself (nb_machine_players == 2)
	// or the machine plays against a human and it is the machine turn
	machine_play(player, row, col);
      }
      
      // Draw the updated board
      draw_board();
      
      // Check if it is possible to color some edges
      color_edges(row, col, player);
      
      // Check if there is a winner or no more possibility
      Path path;
      if(has_won(player, path)) {
	if(player == blue_player)
	  std::cout << "\nBlue Player won! (the b's show a winning path)\n";
	else
	  std::cout << "\nRed Player won! (the r's show a winning path)\n";
	
	// Draw winning path replacing 'B' by 'b' or 'R' by 'r'
	draw_board(&path.route);
	break;
      }
      if(board_is_full())
	throw std::runtime_error{"There is a bug! There must be a winner"};

      // Time for the other player to play
      player = (player == blue_player)? red_player : blue_player;
    }
  }

  /*
   * Each time a position is selected on the board an/some edge(s)
   * between adjacent cells owned by the same player (either Blue or Red) can
   * be colored.
   * e.g. for a 3x3 board:
   *  . - . - B                                    . - . - B 
   *   \ / \ / \                                    \ / \ / \
   *    B - . - .   -> Blue Selects cell (1,1) =>    B - B - . 
   *     \ / \ / \                                    \ / \ / \  
   *      . - . - .                                    . - . - .    
   *
   *  => Edge between node (1, 0) and node (1, 1) and edge between node (1, 1)
   *  and node (0, 2) are given the color Blue.
   */
  void color_edges(unsigned int board_row,
		   unsigned int board_column,
		   unsigned int player_id) {
    // Convert board's row and column to node row and column
    // (that is because I added a layer of 'OUTSIDE' nodes around the board)
    unsigned int node_row{board_row+1};
    unsigned int node_column{board_column+1};
    unsigned int node_id{node_row*(board_size+2) + node_column};
    //std::cout << "node id " << node_id << std::endl;
    std::vector<std::pair<unsigned int, double>> neighbors;
    EdgeColor color{EdgeColor::Any};
    graph.get_neighbors(node_id, neighbors, &color);
    for(auto neighbor: neighbors) {
      unsigned int neighbor_id{neighbor.first};
      if(nodes[neighbor_id].value == static_cast<double>(player_id)) {
	// The adjacent node belongs to the same player
	// Color the edge
	EdgeColor color{static_cast<EdgeColor>(player_id)};
	graph.set_edge_color(nodes[node_id], nodes[neighbor_id], color);
      }
    }
  }

  // For the Blue player check if a path between West and East exist
  // For the Red player check if a path between North and South exist
  bool has_won(unsigned int player, Path& path) {
    
    EdgeColor color{static_cast<EdgeColor>(player)};

    // Store all valid path in a queue
    std::priority_queue<Path, std::vector<Path>, std::greater<Path>> all_path;
    
    if (player == blue_player) {
      // The blue player must build a path between West and East
      
      // For all the nodes on the west side
      for(unsigned int west_node_row{1}; west_node_row < board_size+1;
	    ++west_node_row) {
	unsigned int west_node_column{1};
	unsigned int west_node_id{west_node_row*(board_size+2) +
	    west_node_column};
	Node west_node{west_node_id};
	
	// For all the nodes on the east side
	for(unsigned int east_node_row{1}; east_node_row < board_size+1;
	    ++east_node_row) {
	  unsigned int east_node_column{board_size};
	  unsigned int east_node_id{east_node_row*(board_size+2) +
	      east_node_column};
	  Node east_node{east_node_id};

	  //std::cout << "Check path between " << west_node_id << " and ";
	  //std::cout << east_node_id << std::endl;
	  
	  //Find a path
	  DijkstraShortestPath<ColoredGraph<EdgeColor>> dsp{graph};
	  Path path;
	  path.n1 = west_node;
	  path.n2 = east_node;
	  dsp.get_shortest_path(path, &color);

	  if(path.is_valid())
	    all_path.push(path);
	}
      }
    } else { //player == red_player
      // The red player must build a path between North and South
      
      // For all the nodes on the north side
      for(unsigned int north_node_column{1}; north_node_column < board_size+1;
	    ++north_node_column) {
	unsigned int north_node_row{1};
	unsigned int north_node_id{north_node_row*(board_size+2) +
	    north_node_column};
	Node north_node{north_node_id};
	
	// For all the nodes on the south side
	for(unsigned int south_node_column{1}; south_node_column < board_size+1;
	    ++south_node_column) {
	  unsigned int south_node_row{board_size};
	  unsigned int south_node_id{south_node_row*(board_size+2) +
	      south_node_column};
	  Node south_node{south_node_id};
	  
	  //Find a path
	  DijkstraShortestPath<ColoredGraph<EdgeColor>> dsp{graph};
	  Path path;
	  path.n1 = north_node;
	  path.n2 = south_node;
	  dsp.get_shortest_path(path, &color);

	  if(path.is_valid())
	    all_path.push(path);
	}
      }
    }

    if (!all_path.empty()) {
      // At least 1 path (Select (one of) the shortest one)
      path = all_path.top();
      //path.show();
      return true;
    }
    
    // No path  
    return false;
  }
  
};

#endif //HEX_H
