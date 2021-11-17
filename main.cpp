
#include "hex.h"

int main() {
  
#ifdef _TEST_HEX
  // Play the Hex game

  // Default board's size
  unsigned int board_size{7};
  std::string board_size_str;
  // Default number of human player (can be 0, 1 or 2)
  unsigned int nb_human_player{0};
  std::string nb_human_player_str;

  std::cout << "\n*** Welcome to the Hex Game! ***" << std::endl
	    << "Enter the board's size (just press enter for the default "
	    << "value of " << board_size << "):\n";
  //std::cin.ignore();
  getline(std::cin, board_size_str);
  
  std::cout << "Enter the number of human players (0, 1 or 2)" << std::endl
	    << "(just press enter for the default value of "
	    << nb_human_player << "):\n";
  
  //std::cin.ignore();
  getline(std::cin, nb_human_player_str);
  
  if(!board_size_str.empty())
    board_size = atoi(board_size_str.c_str());
  
  if(!nb_human_player_str.empty())
    nb_human_player = atoi(nb_human_player_str.c_str());
  
  // Start the game
  Hex hex(board_size, nb_human_player);

#endif //_TEST_HEX
}
