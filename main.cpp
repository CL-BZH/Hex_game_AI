
#include <thread>

#include "hex.h"
#include "hex_machine_engine.h"

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

  // Get a board
  HexBoard board(board_size);
  
  // Start the game
  
  if(nb_human_player == 2) {
    // Human against Human
    Hex hex(&board, nb_human_player);
  } else {
    unsigned int threads{std::thread::hardware_concurrency()};
    
    std::cout << "The number of threads will be set to " << threads << std::endl
	      << "(press enter if you don't want to change it." << std::endl
	      << "Else enter the number of threads you want)" << std::endl;
    
    std::string threads_str;
    getline(std::cin, threads_str);
    
    // Set the number of threads if it was given
    if(!threads_str.empty())
      threads = atoi(threads_str.c_str());
    
    if(nb_human_player == 1) {
      // Machine against Human
      HexMachineMcIA machine(board_size, threads);
      Hex hex(&board, nb_human_player, &machine);
    } else {
      // Machine against itself
      //HexMachineDummy blue_machine(board_size);
      //HexMachineDummy red_machine(board_size);
      HexMachineMcIA blue_machine(board_size, threads);
      HexMachineMcIA red_machine(board_size, threads);
      Hex hex(&board, nb_human_player, &blue_machine, &red_machine);
    }
  }

#endif //_TEST_HEX
}
