#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hex.h"
#include "hex_machine_bf.h"
#include "hex_machine_engine.h"
#include "hex_machine_mc.h"
#include "hex_machine_mcts.h"
#include "hex_statistics.h"

// Machine configuration structure
struct MachineConfig
{
  // Map to MachineType enum
  enum class Type
  {
    HUMAN,
    DUMMY,
    BRUTE_FORCE,
    MONTE_CARLO,
    MCTS
  };
  Type type;
  unsigned int threads;
  bool quiet_mode;

  // MCTS-specific
  HexMachineMCTS::SelectionStrategy mcts_strategy;
  unsigned int mcts_iterations_factor;

  // Common
  HexStatistics* stats_ptr;

  MachineConfig(Type t = Type::HUMAN)
      : type(t),
        threads(std::thread::hardware_concurrency()),
        quiet_mode(false),
        mcts_strategy(HexMachineMCTS::SelectionStrategy::VISITS),
        mcts_iterations_factor(200),
        stats_ptr(nullptr)
  {
  }

  // Convert to MachineType
  MachineType to_machine_type() const
  {
    switch (type)
    {
      case Type::DUMMY:
        return MachineType::Dummy;
      case Type::BRUTE_FORCE:
        return MachineType::BruteForce;
      case Type::MONTE_CARLO:
        return MachineType::MonteCarlo;
      case Type::MCTS:
        return MachineType::MCTS;
      case Type::HUMAN:
      default:
        return MachineType::Undefined;  // Human is represented as Undefined in
                                        // Hex
    }
  }
};

// Function to display help information
void show_help(const char* program_name)
{
  std::cout << "Usage: " << program_name << " [OPTIONS]\n";
  std::cout << "Play the Hex game with various configurations.\n\n";
  std::cout << "Options:\n";
  std::cout << "  -s, --size SIZE          Board size (default: 7)\n";
  std::cout << "  -m, --machines CONFIG    Machine configuration (default: "
               "HUMAN,HUMAN)\n";
  std::cout << "                           Examples:\n";
  std::cout << "                           - MC                    (Human vs "
               "Monte Carlo AI)\n";
  std::cout << "                           - MCTS                  (Human vs "
               "MCTS AI)\n";
  std::cout << "                           - DUMMY                 (Human vs "
               "Dummy AI)\n";
  std::cout << "                           - BRUTE_FORCE           (Human vs "
               "Brute Force AI)\n";
  std::cout << "                           - (MC, MCTS)            (MC AI vs "
               "MCTS AI)\n";
  std::cout << "                           - (MCTS{strategy:WIN_RATE}, "
               "MC{threads:4})\n";
  std::cout << "  -t, --threads THREADS    Default threads for AI (default: "
               "hardware_concurrency)\n";
  std::cout
      << "  -g, --games GAMES        Number of games to play (default: 1)\n";
  std::cout << "  -q, --quiet              Quiet mode - disable most output "
               "(statistics only)\n";
  std::cout << "  -r, --random-order       Randomly shuffle player order at "
               "each game\n";
  std::cout << "  -h, --help               Show this help message\n";
  std::cout << "\nMachine Types:\n";
  std::cout << "  HUMAN                    Human player\n";
  std::cout << "  DUMMY                    Dummy AI (random moves)\n";
  std::cout << "  BRUTE_FORCE              Brute Force AI\n";
  std::cout << "  MC                       Monte Carlo AI\n";
  std::cout << "  MCTS                     Monte Carlo Tree Search AI\n";
  std::cout << "\nMCTS Parameters:\n";
  std::cout
      << "  strategy                 VISITS or WIN_RATE (default: VISITS)\n";
  std::cout
      << "  threads                  Number of threads (default: system)\n";
  std::cout << "  iterations_factor        Iterations factor (default: 200)\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << program_name << " -s 11 -m MC\n";
  std::cout << "  " << program_name << " -m \"(MC, MCTS)\" -g 10 -q\n";
  std::cout << "  " << program_name
            << " -m \"MCTS{strategy:WIN_RATE,threads:8}\"\n";
  std::cout << "  " << program_name << " -m \"(DUMMY, BRUTE_FORCE)\"\n";
  std::cout
      << "  " << program_name
      << " -m MC -r           (Random order: Human vs MC or MC vs Human)\n";
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

// Parse key-value pairs from machine configuration
std::map<std::string, std::string> parse_parameters(
    const std::string& param_str)
{
  std::map<std::string, std::string> params;
  std::stringstream ss(param_str);
  std::string pair;

  while (std::getline(ss, pair, ','))
  {
    size_t colon_pos = pair.find(':');
    if (colon_pos != std::string::npos)
    {
      std::string key = pair.substr(0, colon_pos);
      std::string value = pair.substr(colon_pos + 1);
      // Remove whitespace
      key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
      value.erase(std::remove_if(value.begin(), value.end(), ::isspace),
                  value.end());
      params[key] = value;
    }
  }

  return params;
}

// Parse a single machine configuration
MachineConfig parse_single_machine_config(const std::string& config_str,
                                          unsigned int default_threads,
                                          bool quiet_mode,
                                          HexStatistics* stats_ptr)
{
  MachineConfig config;
  config.quiet_mode = quiet_mode;
  config.stats_ptr = stats_ptr;
  config.threads = default_threads;

  // Check if there are parameters
  size_t brace_pos = config_str.find('{');
  std::string machine_type = config_str;
  std::string param_str = "";

  if (brace_pos != std::string::npos)
  {
    machine_type = config_str.substr(0, brace_pos);
    size_t end_brace = config_str.find('}', brace_pos);
    if (end_brace != std::string::npos)
    {
      param_str = config_str.substr(brace_pos + 1, end_brace - brace_pos - 1);
    }
  }

  // Parse machine type
  if (machine_type == "HUMAN")
  {
    config.type = MachineConfig::Type::HUMAN;
  }
  else if (machine_type == "DUMMY")
  {
    config.type = MachineConfig::Type::DUMMY;
  }
  else if (machine_type == "BRUTE_FORCE" || machine_type == "BRUTEFORCE")
  {
    config.type = MachineConfig::Type::BRUTE_FORCE;
  }
  else if (machine_type == "MC" || machine_type == "MONTE_CARLO")
  {
    config.type = MachineConfig::Type::MONTE_CARLO;
  }
  else if (machine_type == "MCTS")
  {
    config.type = MachineConfig::Type::MCTS;
  }
  else
  {
    throw std::runtime_error("Unknown machine type: " + machine_type);
  }

  // Parse parameters if any
  if (!param_str.empty())
  {
    auto params = parse_parameters(param_str);

    for (const auto& [key, value] : params)
    {
      if (key == "threads")
      {
        config.threads = std::stoi(value);
      }
      else if (key == "strategy" && config.type == MachineConfig::Type::MCTS)
      {
        if (value == "WIN_RATE")
        {
          config.mcts_strategy = HexMachineMCTS::SelectionStrategy::WIN_RATE;
        }
        else if (value == "VISITS")
        {
          config.mcts_strategy = HexMachineMCTS::SelectionStrategy::VISITS;
        }
        else
        {
          throw std::runtime_error("Unknown MCTS strategy: " + value);
        }
      }
      else if (key == "iterations_factor" &&
               config.type == MachineConfig::Type::MCTS)
      {
        config.mcts_iterations_factor = std::stoi(value);
      }
      // Note: DUMMY and BRUTE_FORCE don't have parameters in current
      // implementation
    }
  }

  return config;
}

// Parse machine configuration string
std::pair<MachineConfig, MachineConfig> parse_machine_config(
    const std::string& config_str, unsigned int default_threads,
    bool quiet_mode, HexStatistics* stats_ptr)
{
  std::pair<MachineConfig, MachineConfig> configs;

  // Default: human vs human
  if (config_str.empty())
  {
    configs.first.type = MachineConfig::Type::HUMAN;
    configs.second.type = MachineConfig::Type::HUMAN;
    return configs;
  }

  // Single machine: human vs AI
  if (config_str.find('(') == std::string::npos)
  {
    configs.first.type = MachineConfig::Type::HUMAN;
    configs.second = parse_single_machine_config(config_str, default_threads,
                                                 quiet_mode, stats_ptr);
    return configs;
  }

  // Pair of machines: (machine1, machine2)
  if (config_str.front() == '(' && config_str.back() == ')')
  {
    std::string inner = config_str.substr(1, config_str.length() - 2);
    size_t comma_pos = inner.find(',');

    if (comma_pos != std::string::npos)
    {
      std::string first_config = inner.substr(0, comma_pos);
      std::string second_config = inner.substr(comma_pos + 1);

      // Remove whitespace
      first_config.erase(
          std::remove_if(first_config.begin(), first_config.end(), ::isspace),
          first_config.end());
      second_config.erase(
          std::remove_if(second_config.begin(), second_config.end(), ::isspace),
          second_config.end());

      configs.first = parse_single_machine_config(first_config, default_threads,
                                                  quiet_mode, stats_ptr);
      configs.second = parse_single_machine_config(
          second_config, default_threads, quiet_mode, stats_ptr);
      return configs;
    }
  }

  throw std::runtime_error("Invalid machine configuration format: " +
                           config_str);
}

// Function to parse command line arguments
bool parse_arguments(int argc, char* argv[], unsigned int& board_size,
                     std::pair<MachineConfig, MachineConfig>& machine_configs,
                     unsigned int& num_games, bool& quiet_mode,
                     bool& random_order)
{
  // Default values
  board_size = 7;
  num_games = 1;
  quiet_mode = false;
  random_order = false;
  unsigned int default_threads = std::thread::hardware_concurrency();
  std::string machine_config_str = "";

  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help")
    {
      show_help(argv[0]);
      return false;
    }
    else if (arg == "-q" || arg == "--quiet")
    {
      quiet_mode = true;
    }
    else if (arg == "-r" || arg == "--random-order")
    {
      random_order = true;
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
    else if (arg == "-m" || arg == "--machines")
    {
      if (i + 1 < argc)
      {
        machine_config_str = argv[++i];
      }
      else
      {
        std::cerr << "Error: --machines requires a value\n";
        return false;
      }
    }
    else if (arg == "-t" || arg == "--threads")
    {
      if (i + 1 < argc)
      {
        default_threads = std::atoi(argv[++i]);
        if (default_threads < 1)
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

  // Parse machine configuration
  try
  {
    machine_configs = parse_machine_config(machine_config_str, default_threads,
                                           quiet_mode, nullptr);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error parsing machine configuration: " << e.what() << "\n";
    return false;
  }

  return true;
}

// Randomly shuffle the machine configurations
std::pair<MachineConfig, MachineConfig> shuffle_machine_configs(
    const std::pair<MachineConfig, MachineConfig>& configs)
{
  std::pair<MachineConfig, MachineConfig> shuffled = configs;

  // Create a random device and generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dist(0.5);

  // 50% chance to swap the configurations
  if (dist(gen))
  {
    std::swap(shuffled.first, shuffled.second);
  }

  return shuffled;
}

// Create machine instances based on configuration
void create_machines(const std::pair<MachineConfig, MachineConfig>& configs,
                     unsigned int board_size,
                     std::unique_ptr<HexMachineEngine>& blue_machine,
                     std::unique_ptr<HexMachineEngine>& red_machine,
                     unsigned int& nb_human_players)
{
  nb_human_players = 0;

  // Blue player
  switch (configs.first.type)
  {
    case MachineConfig::Type::HUMAN:
      nb_human_players++;
      blue_machine.reset();
      break;
    case MachineConfig::Type::DUMMY:
      blue_machine = std::make_unique<HexMachineDummy>(board_size);
      break;
    case MachineConfig::Type::BRUTE_FORCE:
      blue_machine =
          std::make_unique<HexMachineBF>(board_size, configs.first.threads);
      break;
    case MachineConfig::Type::MONTE_CARLO:
      blue_machine = std::make_unique<HexMachineMcIA>(
          board_size, configs.first.threads, configs.first.quiet_mode,
          configs.first.stats_ptr);
      break;
    case MachineConfig::Type::MCTS:
      blue_machine = std::make_unique<HexMachineMCTS>(
          board_size, configs.first.threads, configs.first.quiet_mode,
          configs.first.stats_ptr, configs.first.mcts_strategy);
      break;
  }

  // Red player
  switch (configs.second.type)
  {
    case MachineConfig::Type::HUMAN:
      nb_human_players++;
      red_machine.reset();
      break;
    case MachineConfig::Type::DUMMY:
      red_machine = std::make_unique<HexMachineDummy>(board_size);
      break;
    case MachineConfig::Type::BRUTE_FORCE:
      red_machine =
          std::make_unique<HexMachineBF>(board_size, configs.second.threads);
      break;
    case MachineConfig::Type::MONTE_CARLO:
      red_machine = std::make_unique<HexMachineMcIA>(
          board_size, configs.second.threads, configs.second.quiet_mode,
          configs.second.stats_ptr);
      break;
    case MachineConfig::Type::MCTS:
      red_machine = std::make_unique<HexMachineMCTS>(
          board_size, configs.second.threads, configs.second.quiet_mode,
          configs.second.stats_ptr, configs.second.mcts_strategy);
      break;
  }
}

// Function to get machine type name
std::string get_machine_type_name(MachineConfig::Type type)
{
  switch (type)
  {
    case MachineConfig::Type::HUMAN:
      return "Human";
    case MachineConfig::Type::DUMMY:
      return "Dummy";
    case MachineConfig::Type::BRUTE_FORCE:
      return "Brute Force";
    case MachineConfig::Type::MONTE_CARLO:
      return "Monte Carlo";
    case MachineConfig::Type::MCTS:
      return "MCTS";
    default:
      return "Unknown";
  }
}

// Function to play a single game
void play_game(unsigned int board_size,
               const std::pair<MachineConfig, MachineConfig>& machine_configs,
               bool quiet_mode, int game_number, int total_games,
               HexStatistics* stats_ptr = nullptr)
{
  if (!quiet_mode && total_games > 1)
  {
    std::cout << "\n=== Game " << game_number << "/" << total_games << " ===\n";
  }

  // Update stats pointers
  auto configs_with_stats = machine_configs;
  configs_with_stats.first.stats_ptr = stats_ptr;
  configs_with_stats.second.stats_ptr = stats_ptr;
  configs_with_stats.first.quiet_mode = quiet_mode;
  configs_with_stats.second.quiet_mode = quiet_mode;

  if (!quiet_mode)
  {
    std::cout << "\n*** Hex Game Configuration ***\n";
    std::cout << "Board size: " << board_size << "x" << board_size << "\n";
    std::cout << "Blue Player: "
              << get_machine_type_name(configs_with_stats.first.type);
    if (configs_with_stats.first.type != MachineConfig::Type::HUMAN)
    {
      std::cout << " (threads: " << configs_with_stats.first.threads << ")";
      if (configs_with_stats.first.type == MachineConfig::Type::MCTS)
      {
        std::cout << " [strategy: "
                  << (configs_with_stats.first.mcts_strategy ==
                              HexMachineMCTS::SelectionStrategy::VISITS
                          ? "VISITS"
                          : "WIN_RATE")
                  << "]";
      }
    }
    std::cout << "\n";

    std::cout << "Red Player:  "
              << get_machine_type_name(configs_with_stats.second.type);
    if (configs_with_stats.second.type != MachineConfig::Type::HUMAN)
    {
      std::cout << " (threads: " << configs_with_stats.second.threads << ")";
      if (configs_with_stats.second.type == MachineConfig::Type::MCTS)
      {
        std::cout << " [strategy: "
                  << (configs_with_stats.second.mcts_strategy ==
                              HexMachineMCTS::SelectionStrategy::VISITS
                          ? "VISITS"
                          : "WIN_RATE")
                  << "]";
      }
    }
    std::cout << "\n";

    if (total_games > 1)
    {
      std::cout << "Game: " << game_number << "/" << total_games << "\n";
    }
    std::cout << "Quiet mode: " << (quiet_mode ? "yes" : "no") << "\n";
    std::cout << "******************************\n\n";
  }

  HexBoard board(board_size);
  std::unique_ptr<HexMachineEngine> blue_machine, red_machine;
  unsigned int nb_human_players;

  create_machines(configs_with_stats, board_size, blue_machine, red_machine,
                  nb_human_players);

  Hex hex(&board, nb_human_players, blue_machine.get(), red_machine.get(),
          quiet_mode, stats_ptr);
}

// Function to play a single game with timing
std::chrono::microseconds play_game_timed(
    unsigned int board_size,
    const std::pair<MachineConfig, MachineConfig>& machine_configs,
    bool quiet_mode, int game_number, int total_games,
    HexStatistics* stats_ptr = nullptr)
{
  auto game_start = std::chrono::high_resolution_clock::now();

  play_game(board_size, machine_configs, quiet_mode, game_number, total_games,
            stats_ptr);

  auto game_end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(game_end -
                                                               game_start);
}

int main(int argc, char* argv[])
{
#ifdef _TEST_HEX
  unsigned int board_size, num_games;
  bool quiet_mode, random_order;
  std::pair<MachineConfig, MachineConfig> machine_configs;
  HexStatistics stats;

  // Parse command line arguments
  if (!parse_arguments(argc, argv, board_size, machine_configs, num_games,
                       quiet_mode, random_order))
  {
    return 1;
  }

  std::vector<std::chrono::microseconds> game_times;
  auto total_start = std::chrono::high_resolution_clock::now();

  // Play games with individual timing
  for (unsigned int game = 1; game <= num_games; ++game)
  {
    auto current_configs = machine_configs;

    // Randomly shuffle order if requested
    if (random_order)
    {
      current_configs = shuffle_machine_configs(machine_configs);
      if (!quiet_mode && num_games > 1)
      {
        std::cout << "Game " << game << " order: "
                  << get_machine_type_name(current_configs.first.type)
                  << " (Blue) vs "
                  << get_machine_type_name(current_configs.second.type)
                  << " (Red)\n";
      }
    }

    auto game_time = play_game_timed(board_size, current_configs, quiet_mode,
                                     game, num_games, &stats);
    game_times.push_back(game_time);

    if (!quiet_mode && num_games > 1)
    {
      auto seconds =
          std::chrono::duration_cast<std::chrono::seconds>(game_time);
      auto microseconds =
          game_time -
          std::chrono::duration_cast<std::chrono::microseconds>(seconds);
      std::cout << "Game " << game << " completed in " << seconds.count()
                << " s " << microseconds.count() << " μs\n";
    }
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      total_end - total_start);

  // Calculate timing statistics
  auto total_game_time = std::chrono::microseconds(0);
  for (const auto& time : game_times)
  {
    total_game_time += time;
  }

  auto avg_game_time = total_game_time / num_games;
  auto min_game_time = *std::min_element(game_times.begin(), game_times.end());
  auto max_game_time = *std::max_element(game_times.begin(), game_times.end());

  auto avg_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(avg_game_time);
  auto avg_microseconds = avg_game_time;
  avg_microseconds -=
      std::chrono::duration_cast<std::chrono::microseconds>(avg_seconds);

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
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       total_game_time)
                       .count()
                << " ms\n";
    }
    else
    {
      std::cout << "Total execution time: "
                << std::chrono::duration_cast<std::chrono::seconds>(
                       total_duration)
                       .count()
                << " s\n";
      std::cout << "Total game time: "
                << std::chrono::duration_cast<std::chrono::seconds>(
                       total_game_time)
                       .count()
                << " s\n";
      std::cout << "Average game time: " << avg_seconds.count() << " s "
                << avg_microseconds.count() << " μs\n";
      std::cout << "Fastest game: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       min_game_time)
                       .count()
                << " ms\n";
      std::cout << "Slowest game: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       max_game_time)
                       .count()
                << " ms\n";
      std::cout << "Games per second: "
                << (num_games * 1000000.0) / total_game_time.count() << "\n";
    }

    if (random_order)
    {
      std::cout << "Player order was randomized for each game\n";
    }
  }

  // Generate CSV filename with configuration info
  std::string date_str = get_current_date();
  std::string blue_type = get_machine_type_name(machine_configs.first.type);
  std::string red_type = get_machine_type_name(machine_configs.second.type);

  // Convert to lowercase for filename and replace spaces with underscores
  std::transform(blue_type.begin(), blue_type.end(), blue_type.begin(),
                 ::tolower);
  std::transform(red_type.begin(), red_type.end(), red_type.begin(), ::tolower);
  std::replace(blue_type.begin(), blue_type.end(), ' ', '_');
  std::replace(red_type.begin(), red_type.end(), ' ', '_');

  std::string csv_filename = "./Data/hex_stats_" + blue_type + "_vs_" +
                             red_type + "_s" + std::to_string(board_size) +
                             "_g" + std::to_string(num_games);

  if (random_order)
  {
    csv_filename += "_random";
  }

  csv_filename += "_" + date_str + ".csv";

  stats.export_to_csv(csv_filename);

  if (!quiet_mode)
  {
    std::cout << "Statistics exported to: " << csv_filename << std::endl;
  }

#endif  //_TEST_HEX

  return 0;
}
