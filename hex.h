#ifndef HEX_H
#define HEX_H

#include <sstream>
#include <string>

#include "hex_board.h"
#include "hex_machine_engine.h"
#include "hex_ui.h"

enum class PlayerType
{
  Human,
  Machine
};

struct Hex
{
  // Build the Hex board with 'size' hexagonal cells. Each cell is represented
  // by a node in a graph. Hence the graph has 'size'*'size' nodes.
  // But I add surrounding cells around the board (see explanations below),
  // so the graph will have (size + 2)^2 nodes.
  // By default the board is of size 11 by 11 and 2 human players play
  // against each other.
  //
  // If the machine plays against itself the two engines must be provided.
  // Then one_machine_engine_ptr will be the engine for the 'Blue'
  // and other_machine_engine_ptr will be the machine for the 'Red'.

  Hex(HexBoard* board_ptr = nullptr, uint nb_players = max_players,
      HexMachineEngine* one_machine_engine_ptr = nullptr,
      HexMachineEngine* other_machine_engine_ptr = nullptr,
      bool quiet_mode = false, HexStatistics* stats_ptr = nullptr)
      : board_ptr{board_ptr},
        nb_human_players{nb_players},
        nb_machine_players{max_players - nb_human_players},
        stats{stats_ptr}
  {
    if (board_ptr == nullptr)
      throw std::runtime_error{"You must provide a board"};

    if (!((nb_players == 0) || (nb_players == 1) || (nb_players == 2)))
      throw std::runtime_error{"The number of players can only be 0, 1 or 2"};

    if (nb_machine_players != 0)
    {
      // At least one machine plays
      if ((nb_machine_players == 1) && (one_machine_engine_ptr == nullptr))
        throw std::runtime_error{
            "You must provide an engine for"
            "the machine to play"};
      if ((nb_machine_players == 2) && ((one_machine_engine_ptr == nullptr) ||
                                        (other_machine_engine_ptr == nullptr)))
        throw std::runtime_error{
            "You must provide two engines for "
            "the machine to play against itself"};

      if (nb_machine_players == 1)
      {
        // Note: the human player will be asked to select its color and
        // then the machine's color will be changed accordingly
        machine_players.push_back({blue_player, one_machine_engine_ptr});
      }
      else
      {
        // If the machine plays against itself the first provided engine
        // is supposed to be for the blue machine and the second one for
        // the red machine.
        machine_players.push_back({blue_player, one_machine_engine_ptr});
        machine_players.push_back({red_player, other_machine_engine_ptr});
      }

      // Set the back pointer
      board_ptr->set_owner(this);
    }

    if (!quiet_mode)
      // Give access to the UI
      board_ptr->set_ui(hex_ui);
#ifdef _DEBUG
    else
      board_ptr->set_ui(hex_ui);
#endif
    // Start the game
    start(quiet_mode);
  }

  std::string get_player_machine_type(uint player_id) const
  {
    for (const auto& machine : machine_players)
    {
      if (machine.color == player_id)
      {
        return machine.engine->get_name();
      }
    }
    return "Unknown";  // Human player
  }

private:
  HexStatistics* stats = nullptr;

  HexUI hex_ui;

  // The board on which to play
  HexBoard* board_ptr;

  // The number of human players
  // 0: the machine plays against itself
  // 1: the machine is the other player
  // 2: 2 players play against each other
  uint nb_human_players;

  // The number of machine players (2 - nb_human_players)
  uint nb_machine_players;

  // Store the id and type of the machine engine (if any)
  struct Machine
  {
    uint color;
    HexMachineEngine* engine;
  };
  std::vector<Machine> machine_players;

  friend HexMachineEngine;

  std::string get_player_name(uint player_color) const
  {
    for (const auto& machine : machine_players)
    {
      if (machine.color == player_color)
      {
        return machine.engine->get_name() + "_" +
               (player_color == blue_player ? "Blue" : "Red");
      }
    }
    return "Human_" + std::string(player_color == blue_player ? "Blue" : "Red");
  }

  // Invite the human player to enter the position he wants on the board
  void prompt_player(uint human_player_id, bool& game_end)
  {
    /*unsigned*/ int row, col;
    std::stringstream ss;

    // Give an example of how to enter a position if it is the first time
    static bool give_example{true};

    // Check if a valid position was selected
    bool position_is_selected{false};

    // Loop until the player selects a valid position on the board
    while (!position_is_selected)
    {
      ss.clear();
      ss << PlayersColors::color(human_player_id)
         << " player selects a position ";

      if (give_example)
      {
        ss << "(Example for row 0 column 1 enter: 0 1 <enter> ):";
        give_example = false;
      }
      ss << std::endl;
      hex_ui << ss.str();

      // Prompt for input values row and column
      hex_ui >> row >> col;

      std::stringstream err;
      // Loop until valid row and column are entered
      position_is_selected =
          board_ptr->select(row, col, human_player_id, game_end, true, &err);

      if (!position_is_selected) hex_ui << err.str();
    }
  }

  // The machine plays against itself or again a human player
  void machine_play(uint machine_color, bool& game_end)
  {
    HexMachineEngine* engine;

    uint row{0}, col{0};

    if ((nb_machine_players == 0) || (nb_machine_players > max_players))
      throw std::runtime_error{
          "Bug: machine_play() called with wrong number "
          "of machine player"};
    else if (nb_machine_players == 1)
      engine = machine_players[0].engine;
    else  // nb_machine_players == 2
      engine = machine_players[machine_color].engine;

    // Check if a valid position was selected
    bool position_is_selected{false};

    // Loop until the machine selects a valid position on the board
    // A loop is needed here for the 'Dummy' machine which tries
    // to get any position randomly
    while (!position_is_selected)
    {
      engine->get_position(*board_ptr, row, col, machine_color);

      position_is_selected =
          board_ptr->select(row, col, machine_color, game_end, true);
    }
  }

  // When there is 1 or 2 human players, 1 of them has to choose its color.
  // The other player (either human or machine) will be given the other color.
  void color_selection()
  {
    uint player;
    uint opponent;
    std::stringstream ss;

    // Invite one of the player to select its color
    ss << "Select your color (Blue starts)" << std::endl;
    ss << PlayersColors::color(blue_player) << " enter " << blue_player << ", ";
    ss << PlayersColors::color(red_player) << " enter " << red_player << "\n";

    std::cout << ss.str();
    // empty ss for reuse
    ss.str(std::string());

    while (true)
    {
      std::cin >> player;
      if ((player == blue_player) || (player == red_player)) break;
      std::cout << "Please enter " << blue_player << " or " << red_player
                << std::endl;
    }

    if (player == blue_player)
    {
      opponent = red_player;
      // p1 selected Blue => p1 starts
      ss << "You selected the color Blue. You start.\n"
         << "To win you must create a path between West and East.\n"
         << "Your opponent must build a path between North and South.\n\n";
    }
    else
    {
      opponent = blue_player;
      ss << "You selected the color Red. Your opponent will start.\n"
         << "To win you must create a path between North and South.\n"
         << "Your opponent must build a path between West and East.\n\n";
    }
    ss << "Press <enter> to start playing";
    std::cout << ss.str();

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // If there is 1 machine player update the color of the machine player
    if (nb_machine_players == 1) machine_players[0].color = opponent;
  }

  // Start the game
  void start(bool quiet_mode)
  {
    // If there is 1 human player against the machine, ask him/her to select its
    // color (the opponent, i.e. the machine will be given the other color).
    if (nb_human_players == 1) color_selection();

    if (!quiet_mode)
    {
      hex_ui.start();

      // Draw the empty board for the human player
      if ((nb_human_players == max_players) ||
          ((nb_human_players == 1) && (machine_players[0].color == red_player)))
      {
        // hex_ui << "Press any key to start\n";
        board_ptr->draw_board();
      }
    }

    // The Blue player always starts
    uint player{blue_player};

    bool game_end{false};

    // Run the game until there is a winner (game_end becomes true)
    // Nota Bene: there is always a winner so there is no need to
    // check if the board is full
    while (true)
    {
      // Invite human player to play if the machine does not play alone.
      // There are 2 cases:
      //     2 humans play against each other
      //     or
      //     1 human plays against the machine and it is his turn
      //     to select a position on the board
      if ((nb_human_players == max_players) ||
          ((nb_human_players == 1) && (player != machine_players[0].color)))
      {
        // Invite the player to select a position
        prompt_player(player, game_end);
      }
      else
      {
        // Either the machine plays against itself (nb_machine_players == 2)
        // or the machine plays against a human and it is the machine's turn
        machine_play(player, game_end);
      }

      if (!quiet_mode)
      {
        if (!game_end)
          // Draw the updated board
          board_ptr->draw_board();
        else
        {
          // Draw winning path replacing 'B' by 'b' or 'R' by 'r'
          board_ptr->draw_board(&(board_ptr->get_path()->route));
          if (player == blue_player)
            hex_ui.print("\nBlue won! (the b's show the winning path)\n");
          else
            hex_ui.print("\nRed won! (the r's show the winning path)\n");

          // Show the path as a list of node
          std::cout << "Path:" << std::endl;
          for (auto node : board_ptr->get_path()->route)
          {
            uint row, col;
            if (board_ptr->get_onboard_row_column(node.id, row, col))
              std::cout << " (" << row << ", " << col << ") ->";
            else
            {
              std::stringstream err;
              err << "Invalid node id for a cell on the board." << std::endl;
              throw std::runtime_error{err.str()};
            }
          }
          std::cout << "\b\b  " << std::endl;
        }
      }

      if (game_end)
      {
        if (stats)
        {
          std::string winner_name = get_player_name(player);
          std::string loser_name =
              get_player_name(player == blue_player ? red_player : blue_player);
          stats->record_game_result(winner_name, loser_name);
        }
        break;
      }

      // Now it is the turn of the other player to play
      player = (player == blue_player) ? red_player : blue_player;

      if (board_ptr->board_is_full())
        throw std::runtime_error("Board is full!!!");
    }
  }
};

inline std::string HexBoard::get_player_machine_type(uint player_id) const
{
  if (!owner_hex) return "Unknown";
  return owner_hex->get_player_machine_type(player_id);
}

#endif  // HEX_H
