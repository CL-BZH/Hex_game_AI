#ifndef HEX_BOARD_H
#define HEX_BOARD_H

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


// Maximun number of players
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

const unsigned int min_board_size{3};
const unsigned int max_board_size{99}; //This is very unrealistic...

struct HexBoard {

  HexBoard(unsigned int size=11):
    board_size(size),
    board_cells(size*size),
    graph_size{(size+2)*(size+2)},
    graph{graph_size, EdgeColor::Any} {
      
      if (size < min_board_size) {
	std::stringstream err;
	err << "The board's size should be at least " << min_board_size;
	throw std::runtime_error{err.str()};
      }
      if (size > max_board_size) {
	std::stringstream err;
	err << "The maximum board's size is " << max_board_size;
	throw std::runtime_error{err.str()};
      }
	
      // Initialize the nodes
      init_nodes();
      
      // Build the graph that represent the Hex game (i.e. add edges)
      build_graph();
    }

  // Copy constructor
  HexBoard(const HexBoard& rhs): board_size{rhs.board_size},
				 board_cells{rhs.board_cells},
				 graph_size{rhs.graph_size},
				 nb_selected_cells{rhs.nb_selected_cells},
				 graph{graph_size, EdgeColor::Any} {

    // Copy the nodes vector
    nodes = rhs.nodes;
    
    // Copy the original graph in this one
    copy_graph(rhs.graph);
  }

  void release_board_node(unsigned int node_id) {
    // Clear edges color
    uncolor_edges(node_id);

    // release the node
    set_node_value(node_id, NodeValue::AVAILABLE);
      
    // Decrease by 1 the total number of selected cells
    --nb_selected_cells;
  }

  void release_board_node(unsigned int board_row, unsigned int board_column) {
    unsigned int node_row{board_row+1};
    unsigned int node_column{board_column+1};
    unsigned int node_id{node_row*(board_size+2) + node_column};
    release_board_node(node_id);
  }
  
  void release_board_nodes(std::vector<unsigned int>& nodes_ids) {
    for(auto node_id: nodes_ids) {
      release_board_node(node_id);
    }
  }
  
  void fill_with_color(unsigned int player_id,
		       const std::vector<std::array<unsigned int, 2>>& cells) {
    unsigned int node_row;
    unsigned int node_column;
    unsigned int node_id;
    
    for(auto cell: cells) {
      node_row = cell[0] + 1;
      node_column = cell[1] + 1;
      node_id = node_row*(board_size+2) + node_column;
      // Register the player has owner of this selected cell
      nodes[node_id].value = static_cast<double>(player_id);
    
      // Increase by 1 the total number of selected cells
      ++nb_selected_cells;

      // Check if it is possible to color some edges
      color_edges(node_id, player_id);
    }
    //draw_board();
  }

  void complete_with_color(unsigned int player_id) {
    
    unsigned int node_id;
    std::vector<unsigned int> available_nodes_ids;
    
    // Get index of all available positions
    for(unsigned int node_row{1}; node_row < board_size+1; ++node_row) {
      for(unsigned int node_column{1}; node_column < board_size+2; ++node_column) {
	node_id = node_row*(board_size+2) + node_column;
	if(is_node_available(node_id)) {
	  //std::cout << "Available: " << node_id << std::endl;
	  available_nodes_ids.push_back(node_id);
	}
      }
    }
    
    //std::cout << "Available nodes: " << available_nodes_ids.size() << std::endl;
    
    for(auto node_id: available_nodes_ids) {
      // Register the player has owner of this selected cell
      nodes[node_id].value = static_cast<double>(player_id);
    
      // Increase by 1 the total number of selected cells
      ++nb_selected_cells;

      // Check if it is possible to color some edges
      color_edges(node_id, player_id);
    }
    //draw_board();
  }

  void rand_fill_board(unsigned int first_player_id, std::mt19937& gen,
		       std::vector<unsigned int>& available_nodes_ids) {
    
    unsigned int node_id;
      
    // Get index of all available positions
    for(unsigned int node_row{1}; node_row < board_size+1; ++node_row) {
      for(unsigned int node_column{1}; node_column < board_size+2; ++node_column) {
	node_id = node_row*(board_size+2) + node_column;
	if(is_node_available(node_id)) {
	  //std::cout << "Available: " << node_id << std::endl;
	  available_nodes_ids.push_back(node_id);
	}
      }
    }
    
    //std::cout << "Available nodes: " << available_nodes_ids.size() << std::endl;

    //Shuffle available node_id
    std::shuffle(std::begin(available_nodes_ids),
		 std::end(available_nodes_ids), gen);

    unsigned int player_id{first_player_id};
    
    for(auto node_id: available_nodes_ids) {
      // Register the player has owner of this selected cell
      nodes[node_id].value = static_cast<double>(player_id);
    
      // Increase by 1 the total number of selected cells
      ++nb_selected_cells;

      // Check if it is possible to color some edges
      color_edges(node_id, player_id);

      // switch players
      player_id = (player_id == blue_player)? red_player : blue_player;
    }
    //draw_board();
  }

  
  // Select a position on the board.
  // Returns 'true' if a valid position was selected (false otherwise)
  bool select(unsigned int board_row,
	      unsigned int board_column,
	      unsigned int player_id,
	      bool& game_end,
	      bool draw=false,
	      std::stringstream* err=nullptr) {

    if(game_end)
      throw std::runtime_error{"Cannot call select with game_end == true"};     
      
    if (!((player_id == blue_player) || (player_id == red_player)))
      throw std::runtime_error{"Invalid player's Id"};     
    
    // Convert board's row and column to node row and column
    // (that is because I added a layer of 'OUTSIDE' nodes around the board)
    unsigned int node_row{board_row+1};
    unsigned int node_column{board_column+1};
    unsigned int node_id{node_row*(board_size+2) + node_column};
    NodeValue node_value;
    
    if((board_row >= board_size) || (board_column >= board_size) ||
       (!is_on_board(node_id))) {
      if(err != nullptr)
	*err << "Not valid row " << board_row << " or column "
	     << board_column << std::endl;
      return false;
    }

    node_value= static_cast<NodeValue>(nodes[node_id].value);
    
    if (node_value != NodeValue::AVAILABLE) {
      if(err != nullptr)
	*err << "Already selected position";
      return false;
    }
    
    // From here select returns true (i.e. a position was selected)
    
    // Register the player has owner of this selected cell
    nodes[node_id].value = static_cast<double>(player_id);
    
    // Increase by 1 the total number of selected cells
    ++nb_selected_cells;

    // Check if it is possible to color some edges
    color_edges(node_id, player_id);
    
    // Check if there is a winner or no more possibility
    Path path;
    if(has_won(player_id, &path)) {
      // End the game
      game_end = true;
      
      if(draw) {
	std::stringstream sstr;
	if(player_id == blue_player)
	  sstr << "\nBlue won! (the b's show the winning path)" << std::endl;
	else
	  sstr << "\nRed won! (the r's show the winning path)" << std::endl;

	if(ui != nullptr)
	  ui->print(sstr.str());
	else
	  std::cout << sstr.str();
	
	// Draw winning path replacing 'B' by 'b' or 'R' by 'r'
	draw_board(&path.route);
      }
    }

    // Position selected = true
    return true;
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

    // Draw column indexes
    for(unsigned int column{0}; column < board_size; ++column) {
      if(column < 10)
	ss << std::string(3, ' ') << column;
      else
	ss << std::string(2, ' ') << column;
    }

    ss << std::endl;

    unsigned int row{1};
    
    for(; row < board_size+1; ++row) {
  
      // Draw row index
      if(row < 11)
	ss << std::string(2*row - 1, ' ') << row - 1 << std::string(1, ' ');
      else // row - 1 will be bigger than 9 (and less than 100 by construction)
	ss << std::string(2*row - 2, ' ') << row - 1 << std::string(1, ' ');
      
      // Draw the nodes' content ('.', 'R' or 'B') and West-East edges
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
      // Write the row index again on the right of the board
      ss << std::string(1, ' ') << row - 1;
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
    
    // Draw column indexes again at the bottom
    ss << std::string(2*row - 1, ' ');
    for(unsigned int column{0}; column < board_size; ++column) {
      if(column < 10)
	ss << std::string(3, ' ') << column;
      else
	ss << std::string(2, ' ') << column;
	
    }
    ss << std::endl;

    std::string str{ss.str()};
    
    if(ui != nullptr)
      ui->draw_board(str);
    else
      std::cout << str;

  }

  // Tells if the board is full
  bool board_is_full() const {
    return (nb_selected_cells == board_cells);
  }

  // Tells if the board is empty
  bool board_is_empty() const {
    return (nb_selected_cells == 0);
  }

  unsigned int nb_available_cells() const {
    return board_cells - nb_selected_cells;
  }
  
  // Get the percentage of available cells on the board
  double percentage_available_cells() const {
    return (static_cast<double>(nb_available_cells())) / board_cells;
  }

  // Get the first cell availalable starting from a given row and column
  bool get_first_available_position(unsigned int& board_row,
				    unsigned int& board_column,
				    unsigned int player_id,
				    bool& game_end) {

    for(; board_row < board_size; ++board_row) {
      for(; board_column < board_size; ++board_column) {
	if(select(board_row, board_column, player_id, game_end))
	   return true;
	// Try next column in the current line
      }
      board_column = 0;
      // Try next line
    }
    
    return false;
  }
  
  void get_all_available_position(std::vector<std::array<unsigned int, 2>>&
				  positions) {

    bool game_end{false};
    unsigned int board_row{0};
    unsigned int board_column{0};
      
    for(; board_row < board_size; ++board_row) {
      for(; board_column < board_size; ++board_column) {
	// force "game_end" to false
	game_end = false;
	if(get_first_available_position(board_row, board_column, 0, game_end))
	  positions.push_back({board_row, board_column});
	// Try next column in the current line
      }
      board_column = 0;
      // Try next line
    }
  }

  /*
   * Check if a player identified by its 'player_id' has won the game, that is:
   *  - for the Blue player check if a path between West and East exist
   *  - for the Red player check if a path between North and South exist
   * When the 'quality' pointer is not null it also computes a quality criteria.
   * The 'quality' is defined as the sum of the inverse of path length for all
   * shortest path that lead to a win.
   * Of course if 'has_won()' is called after each player's move there can be
   * at the very most 1 winning path. But if the function is called with a fully
   * filled board for example (see 'mc_task()' in hex_machine_engine.h) then
   * there can be more than 1 shortest path from one side of the board to the
   * other (of course if there is a path from West to East there cannot be a
   * path from North to South and vice-versa).
   */
  bool has_won(unsigned int player, Path* path=nullptr,
	       double* quality=nullptr) {
    
    EdgeColor color{static_cast<EdgeColor>(player)};

    // Store all valid path in a queue
    std::priority_queue<Path, std::vector<Path>, std::greater<Path>> all_path{};
    
    if (player == blue_player) {
      // The blue player must build a path between West and East
      
      // For all the nodes on the west side check if there is/are a path(s)
      // to east nodes (i.e. nodes on the last column)
      const unsigned int west_node_column{1};
      for(unsigned int west_node_row{1}; west_node_row < board_size+1;
	    ++west_node_row) {
	unsigned int west_node_id{west_node_row*(board_size+2) +
	    west_node_column};
	
	// If the cell is not blue then pass since there is no way to start
	// a blue path from that node
	if(get_node_value(west_node_id) != NodeValue::B)
	  continue;
	
	Node west_node{west_node_id};

	// For all the nodes on the east side check if there is a path from
	// the current west node to that east node
	const unsigned int east_node_column{board_size};
	for(unsigned int east_node_row{1}; east_node_row < board_size+1;
	    ++east_node_row) {
	  unsigned int east_node_id{east_node_row*(board_size+2) +
	      east_node_column};
	  
	  // If the cell is not blue then pass since there is no way to end
	  // a blue path from that node
	  if(get_node_value(east_node_id) != NodeValue::B)
	    continue;
	  
	  Node east_node{east_node_id};

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
      
      // For all the nodes on the north side check if there is/are a path(s)
      // to south nodes (i.e. nodes on the bottom row)
      unsigned int north_node_row{1};
      for(unsigned int north_node_column{1}; north_node_column < board_size+1;
	    ++north_node_column) {
	unsigned int north_node_id{north_node_row*(board_size+2) +
	    north_node_column};

	// If the cell is not red then pass since there is no way to start
	// a red path from that node
	if(get_node_value(north_node_id) != NodeValue::R)
	  continue;
	
	Node north_node{north_node_id};
	
	// For all the nodes on the south side check if there is a path from
	// the current north node to that south node
	const unsigned int south_node_row{board_size};
	for(unsigned int south_node_column{1}; south_node_column < board_size+1;
	    ++south_node_column) {
	  unsigned int south_node_id{south_node_row*(board_size+2) +
	      south_node_column};
	  
	  // If the cell is not red then pass since there is no way to end
	  // a red path from that node
	  if(get_node_value(south_node_id) != NodeValue::R)
	    continue;
	  
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

    // If all_path.empty() then has_won() == false
    bool did_win{!all_path.empty()};
    
    if ((path != nullptr) && (did_win)) {
      // At least 1 path (Select (one of) the shortest one)
      *path = all_path.top();
      //path.show();
    }

    // Compute the quality of a selection base on the number of shortest
    // path that was obtained. For each shortest path found, its quality
    // is the inverse of its length
    if(quality != nullptr) {
      *quality = 0.0;
      if(did_win) {
	while(!all_path.empty()) {
	  double distance{all_path.top().distance};
	  all_path.pop();
	  *quality += 1 / distance;
	}
      }
    }
    
    return did_win;
  }

  // Get node's value (i.e. Red, Blue, Available, Outside)
  NodeValue get_node_value(unsigned int node_id) {
    return static_cast<NodeValue>(nodes[node_id].value);
  }
  
  void set_ui(HexUI& hex_ui) {
    ui = &hex_ui;
  }
  
private:
  HexUI* ui{nullptr};
  
  // The board size (number of cell per edge. 11 by default)
  unsigned int board_size{11};

  // The total number of cells on the board: board_size*board_size
  unsigned int board_cells;
  
  // The graph size
  unsigned int graph_size;

  // number of selected cells
  unsigned int nb_selected_cells{0};
  
  enum class EdgeColor {
    Blue,
    Red,
    Any,
  };

  //Ncurses nc;
  
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

  // Tell if a node is available
  bool is_node_available(unsigned int node_id) {
    return nodes[node_id].value == static_cast<double>(NodeValue::AVAILABLE);
  }
  
  // Tell if a node is on the board or surrounding it
  bool is_on_board(unsigned int node_id) {
    return nodes[node_id].value != static_cast<double>(NodeValue::OUTSIDE);
  }
  bool is_on_board(unsigned int node_row, unsigned int node_column) {
    unsigned int node_id{node_row*(board_size+2) + node_column};
    return is_on_board(node_id);
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

  void copy_graph(const ColoredGraph<EdgeColor>& graph_to_copy) {
    build_graph();
    graph.set_edges_colors(graph_to_copy.get_edges_colors());
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
  void color_edges(unsigned int node_id,
		   unsigned int player_id) {
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

  void uncolor_edges(unsigned int node_id) {
    //std::cout << "node id " << node_id << std::endl;
    std::vector<std::pair<unsigned int, double>> neighbors;

    graph.get_neighbors(node_id, neighbors);
    for(auto neighbor: neighbors) {
      unsigned int neighbor_id{neighbor.first};
      // Uncolor the edge
      graph.set_edge_color(nodes[node_id], nodes[neighbor_id], EdgeColor::Any);
    }
  }

};


#endif //HEX_BOARD_H
