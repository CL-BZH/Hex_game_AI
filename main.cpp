#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hex.h"
#include "hex_machine_engine.h"
#include "hex_machine_mc.h"
#include "hex_machine_mcts.h"
#include "hex_statistics.h"

// Function to display help information
void show_help(const char* program_name)
{
  std::cout << "Usage: " << program_name << " [OPTIONS]\n";
  std::cout << "Play the Hex game with various configurations.\n\n";
  std::cout << "Options:\n";
  std::cout << "  -s, --size SIZE          Board size (default: 7)\n";
  std::cout << "  -p, --players PLAYERS    Number of human players: 0, 1, or 2 (default: 0)\n";
  std::cout
      << "  -t, --threads THREADS    Number of threads for AI (default: hardware_concurrency)\n";
  std::cout << "  -g, --games GAMES        Number of games to play (default: 1)\n";
  std::cout << "  -q, --quiet              Quiet mode - disable most output (statistics only)\n";
  std::cout << "  -h, --help               Show this help message\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << program_name
            << " -s 11 -p 1 -t 4      # Human vs AI on 11x11 board, 4 threads\n";
  std::cout << "  " << program_name << " -p 0 -g 10 -q        # 10 AI vs AI games, quiet mode\n";
  std::cout << "  " << program_name << " --size 9 --players 2 # Human vs Human on 9x9 board\n";
}

// Function to get current date as string in YYYY-MM-DD format
std::string get_current_date()
{
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm tm = *std::localtime(&time_t);

  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d");
  return ss.str();
}

// Function to parse command line arguments
bool parse_arguments(int argc, char* argv[], unsigned int& board_size,
                     unsigned int& nb_human_player, unsigned int& threads, unsigned int& num_games,
                     bool& quiet_mode)
{
  // Default values
  board_size = 7;
  nb_human_player = 0;
  threads = std::thread::hardware_concurrency();
  num_games = 1;
  quiet_mode = false;

  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help")
    {
      show_help(argv[0]);
      return false;  // Exit after showing help
    }
    else if (arg == "-q" || arg == "--quiet")
    {
      quiet_mode = true;
    }
    else if (arg == "-s" || arg == "--size")
    {
      if (i + 1 < argc)
      {
        board_size = std::atoi(argv[++i]);
        if (board_size < 3)
        {
          std::cerr << "Error: Board size must be at least 3\n";
          return false;
        }
      }
      else
      {
        std::cerr << "Error: --size requires a value\n";
        return false;
      }
    }
    else if (arg == "-p" || arg == "--players")
    {
      if (i + 1 < argc)
      {
        nb_human_player = std::atoi(argv[++i]);
        if (nb_human_player > 2)
        {
          std::cerr << "Error: Number of players must be 0, 1, or 2\n";
          return false;
        }
      }
      else
      {
        std::cerr << "Error: --players requires a value\n";
        return false;
      }
    }
    else if (arg == "-t" || arg == "--threads")
    {
      if (i + 1 < argc)
      {
        threads = std::atoi(argv[++i]);
        if (threads < 1)
        {
          std::cerr << "Error: Number of threads must be at least 1\n";
          return false;
        }
      }
      else
      {
        std::cerr << "Error: --threads requires a value\n";
        return false;
      }
    }
    else if (arg == "-g" || arg == "--games")
    {
      if (i + 1 < argc)
      {
        num_games = std::atoi(argv[++i]);
        if (num_games < 1)
        {
          std::cerr << "Error: Number of games must be at least 1\n";
          return false;
        }
      }
      else
      {
        std::cerr << "Error: --games requires a value\n";
        return false;
      }
    }
    else
    {
      std::cerr << "Error: Unknown option '" << arg << "'\n";
      std::cerr << "Use '" << argv[0] << " --help' for usage information\n";
      return false;
    }
  }

  return true;
}

// Function to play a single game
void play_game(unsigned int board_size, unsigned int nb_human_player, unsigned int threads,
               bool quiet_mode, int game_number, int total_games,
               HexStatistics* stats_ptr = nullptr)
{
  if (!quiet_mode && total_games > 1)
  {
    std::cout << "\n=== Game " << game_number << "/" << total_games << " ===\n";
  }

  if (!quiet_mode)
  {
    std::cout << "\n*** Hex Game Configuration ***\n";
    std::cout << "Board size: " << board_size << "x" << board_size << "\n";
    std::cout << "Human players: " << nb_human_player << "\n";
    std::cout << "AI threads: " << threads << "\n";
    if (total_games > 1)
    {
      std::cout << "Game: " << game_number << "/" << total_games << "\n";
    }
    std::cout << "Quiet mode: " << (quiet_mode ? "yes" : "no") << "\n";
    std::cout << "******************************\n\n";
  }

  HexBoard board(board_size);

  if (nb_human_player == 2)
  {
    // Human against Human
    Hex hex(&board, nb_human_player, nullptr, nullptr, quiet_mode);
  }
  else
  {
    // Randomly decide which machine starts first for AI vs AI
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    bool mcts_starts = dis(gen) == 0;

    if (nb_human_player == 1)
    {
      // Machine against Human
      HexMachineMcIA machine(board_size, threads);
      Hex hex(&board, nb_human_player, &machine, nullptr, quiet_mode);
    }
    else
    {
      // Machine against itself
      if (mcts_starts)
      {
        if (!quiet_mode)
        {
          std::cout << "Blue machine (Monte Carlo Tree Search) starts first!\n";
        }
        HexMachineMCTS blue_machine(board_size, threads, quiet_mode, stats_ptr);
        HexMachineMcIA red_machine(board_size, threads, quiet_mode, stats_ptr);
        Hex hex(&board, nb_human_player, &blue_machine, &red_machine, quiet_mode, stats_ptr);
      }
      else
      {
        if (!quiet_mode)
        {
          std::cout << "Blue machine (Monte Carlo) starts first!\n";
        }
        HexMachineMcIA blue_machine(board_size, threads, quiet_mode, stats_ptr);
        HexMachineMCTS red_machine(board_size, threads, quiet_mode, stats_ptr);
        Hex hex(&board, nb_human_player, &blue_machine, &red_machine, quiet_mode, stats_ptr);
      }
    }
  }
}

// Function to play a single game with timing
std::chrono::microseconds play_game_timed(unsigned int board_size, unsigned int nb_human_player,
                                          unsigned int threads, bool quiet_mode, int game_number,
                                          int total_games, HexStatistics* stats_ptr = nullptr)
{
  auto game_start = std::chrono::high_resolution_clock::now();

  play_game(board_size, nb_human_player, threads, quiet_mode, game_number, total_games, stats_ptr);

  auto game_end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(game_end - game_start);
}

int main(int argc, char* argv[])
{
#ifdef _TEST_HEX
  unsigned int board_size, nb_human_player, threads, num_games;
  bool quiet_mode;
  HexStatistics stats;

  // Parse command line arguments
  if (!parse_arguments(argc, argv, board_size, nb_human_player, threads, num_games, quiet_mode))
  {
    return 1;  // Error exit
  }

  std::vector<std::chrono::microseconds> game_times;
  auto total_start = std::chrono::high_resolution_clock::now();

  // Play games with individual timing
  for (unsigned int game = 1; game <= num_games; ++game)
  {
    auto game_time =
        play_game_timed(board_size, nb_human_player, threads, quiet_mode, game, num_games, &stats);
    game_times.push_back(game_time);

    if (!quiet_mode && num_games > 1)
    {
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(game_time);
      auto microseconds =
          game_time - std::chrono::duration_cast<std::chrono::microseconds>(seconds);
      std::cout << "Game " << game << " completed in " << seconds.count() << " s "
                << microseconds.count() << " μs\n";
    }
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);

  // Calculate timing statistics
  auto total_game_time = std::chrono::microseconds(0);
  for (const auto& time : game_times)
  {
    total_game_time += time;
  }

  auto avg_game_time = total_game_time / num_games;
  auto min_game_time = *std::min_element(game_times.begin(), game_times.end());
  auto max_game_time = *std::max_element(game_times.begin(), game_times.end());

  auto avg_seconds = std::chrono::duration_cast<std::chrono::seconds>(avg_game_time);
  auto avg_microseconds = avg_game_time;
  avg_microseconds -= std::chrono::duration_cast<std::chrono::microseconds>(avg_seconds);

  if (!quiet_mode && num_games > 1)
    std::cout << "\n*** All " << num_games << " games completed! ***\n";

  stats.print_summary();

  if (!quiet_mode)
  {
    // Enhanced timing information
    std::cout << "\nTIMING ANALYSIS:\n";
    if (num_games == 1)
    {
      std::cout << "Game time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(total_game_time).count()
                << " ms\n";
    }
    else
    {
      std::cout << "Total execution time: "
                << std::chrono::duration_cast<std::chrono::seconds>(total_duration).count()
                << " s\n";
      std::cout << "Total game time: "
                << std::chrono::duration_cast<std::chrono::seconds>(total_game_time).count()
                << " s\n";
      std::cout << "Average game time: " << avg_seconds.count() << " s " << avg_microseconds.count()
                << " μs\n";
      std::cout << "Fastest game: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(min_game_time).count()
                << " ms\n";
      std::cout << "Slowest game: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(max_game_time).count()
                << " ms\n";
      std::cout << "Games per second: " << (num_games * 1000000.0) / total_game_time.count()
                << "\n";
    }
  }

  // Generate CSV filename with date
  std::string date_str = get_current_date();
  std::string csv_filename = "./Data/hex_stats_s_" + std::to_string(board_size) + "_t_" +
                             std::to_string(threads) + "_g_" + std::to_string(num_games) + "_" +
                             date_str + ".csv";

  stats.export_to_csv(csv_filename);

  if (!quiet_mode)
  {
    std::cout << "Statistics exported to: " << csv_filename << std::endl;
  }

#endif  //_TEST_HEX

  return 0;
}
