#ifndef HEX_STATISTICS_H
#define HEX_STATISTICS_H

class HexStatistics
{
private:
  struct EngineStats
  {
    std::string name;
    std::vector<std::chrono::microseconds> move_times;
    unsigned int wins{0};
    unsigned int losses{0};
    unsigned int total_moves{0};

    void record_move(std::chrono::microseconds time)
    {
      move_times.push_back(time);
      total_moves++;
    }

    void record_game_result(bool won)
    {
      if (won)
        wins++;
      else
        losses++;
    }

    void print() const
    {
      if (move_times.empty()) return;

      auto total_time = std::accumulate(move_times.begin(), move_times.end(),
                                        std::chrono::microseconds(0));
      auto avg_time = total_time / move_times.size();

      auto sec{std::chrono::duration_cast<std::chrono::seconds>(avg_time)};
      auto us{avg_time -
              std::chrono::duration_cast<std::chrono::microseconds>(sec)};

      double win_rate =
          (wins + losses > 0)
              ? (static_cast<double>(wins) / (wins + losses)) * 100.0
              : 0.0;

      std::cout << name << ": " << wins << " wins, " << losses << " losses "
                << "(" << std::fixed << std::setprecision(1) << win_rate
                << "%) "
                << "Avg move: " << sec.count() << "s" << us.count() << "μs "
                << "[" << total_moves << " moves]" << std::endl;
    }
  };

  std::map<std::string, EngineStats> engines;
  unsigned int total_games{0};
  std::string last_winner;  // Track who won the last game

public:
  // Record the time taken for a move by a specific engine
  void record_move_time(const std::string& engine_name,
                        std::chrono::microseconds duration)
  {
    engines[engine_name].name = engine_name;
    engines[engine_name].record_move(duration);
  }

  // Record the final game result (call this when game ends)
  void record_game_result(const std::string& winner_engine_name,
                          const std::string& loser_engine_name)
  {
    engines[winner_engine_name].record_game_result(true);
    engines[loser_engine_name].record_game_result(false);
    total_games++;
    last_winner = winner_engine_name;
  }

  // Record game result when both engines are the same type
  void record_game_result(const std::string& engine_name, bool won)
  {
    engines[engine_name].record_game_result(won);
    total_games++;
    if (won) last_winner = engine_name;
  }

  void print_summary() const
  {
    std::cout << "\n=== HEX GAME STATISTICS ===" << std::endl;
    std::cout << "Total games: " << total_games << std::endl;

    for (const auto& [name, stats] : engines)
    {
      stats.print();
    }
  }

  void clear()
  {
    engines.clear();
    total_games = 0;
    last_winner.clear();
  }
  void export_to_csv(const std::string& filename) const
  {
    std::ofstream file(filename);
    file << "EngineType,StartingPosition,Wins,Losses,WinRate,LossRate,"
            "TotalMoves,"
            "AvgMoveTimeMicroseconds\n";

    for (const auto& [full_name, stats] : engines)
    {
      std::string engine_type;
      std::string starting_position;

      size_t underscore_pos = full_name.find('_');
      if (underscore_pos != std::string::npos)
      {
        engine_type = full_name.substr(0, underscore_pos);
        starting_position = full_name.substr(underscore_pos + 1);
      }
      else
      {
        engine_type = full_name;
        starting_position = "Unknown";
      }

      double win_rate = (stats.wins + stats.losses > 0)
                            ? (static_cast<double>(stats.wins) /
                               (stats.wins + stats.losses)) *
                                  100.0
                            : 0.0;
      double loss_rate = 100.0 - win_rate;

      std::chrono::microseconds avg_time{0};
      if (!stats.move_times.empty())
      {
        auto total_time =
            std::accumulate(stats.move_times.begin(), stats.move_times.end(),
                            std::chrono::microseconds(0));
        avg_time = total_time / stats.move_times.size();
      }

      file << engine_type << "," << starting_position << "," << stats.wins
           << "," << stats.losses << "," << std::fixed << std::setprecision(2)
           << win_rate << "," << std::fixed << std::setprecision(2) << loss_rate
           << "," << stats.total_moves << "," << avg_time.count() << "\n";
    }
  }
};

#endif  // HEX_STATISTICS_H
