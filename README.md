# Hex_game_AI
This repository contains code I wrote for the course [C++ For C Programmers, Part B](https://www.coursera.org/learn/c-plus-plus-b) delivered by the University of California Santa Cruz on Coursera.

The purpose is to implement the [Hex game](https://en.wikipedia.org/wiki/Hex_(board_game)) where two human players can play against each other or a human plays against the machine or the machine plays against itself.

For the machine to play, I implemented 4 methods in what I call machine engines:  
The engines are named:
- `HexMachineDummy` - Random moves
- `HexMachineBF` - Brute-force search
- `HexMachineMcIA` - Monte Carlo simulation
- `HexMachineMCTS` - Monte Carlo Tree Search

> Examples are given in `main.cpp` on how to select the machine engine(s).

I will explain each of them below in the **Code explanations** section.

## Compilation
### Clean
```bash
make clean
```

### Build
```bash
make
```

### Options
In the Makefile:  
If you wish to use ncurses instead of the default terminal uncomment the lines:  
```bash
#CPPFLAGS += -D_NCURSES  
#LIBS += -lncurses  
```

If you want to enable brute-force fallback for Monte Carlo when few positions remain, uncomment:
```bash
#CPPFLAGS += -D_BF_SUP=20
```

Here are examples of 6x6 boards at the end of some games without and with ncurses.

#### Without ncurses
```text
Blue Player won! (the b's show the winning path)
   0   1   2   3   4   5   6
 0 R - B - B - R - R - b - b 0
    \ / \ / \ / \ / \ / \ / \
   1 R - B - B - R - b - R - . 1
      \ / \ / \ / \ / \ / \ / \
     2 B - . - R - R - b - R - R 2
        \ / \ / \ / \ / \ / \ / \
       3 R - . - b - b - R - B - R 3
          \ / \ / \ / \ / \ / \ / \
         4 . - b - . - . - R - . - B 4
            \ / \ / \ / \ / \ / \ / \
           5 . - b - R - B - R - B - R 5
              \ / \ / \ / \ / \ / \ / \
             6 b - . - R - R - . - B - B 6
                0   1   2   3   4   5   6
```

#### With ncurses
![ncurses_board_example](./Images/ncurses_board_example.png "With ncurses")

> **Note**:
The code for the **User Interface** is in `hex_ui.h`.

## To Play
> See examples of settings in `main.cpp`.  

Call `./hex` once compilation is done.  
You will be invited to enter the size of the board, the number of human players and your color if you want to play against the machine.  

Below is an example:
```text
./hex

*** Hex Game Configuration ***
Board size: 7x7
Human players: 0
AI threads: 8
Quiet mode: no
******************************

Blue machine (Monte Carlo Tree Search) starts first!
Blue MCTS: Running 3302 iterations in 8 threads
Duration : 48 s 498123 us
Blue MCTS: Selected move (2,5) - Win rate: 59% (77 visits)
   0   1   2   3   4   5   6
 0 . - . - . - . - . - . - . 0
    \ / \ / \ / \ / \ / \ / \
   1 . - . - . - . - . - . - . 1
      \ / \ / \ / \ / \ / \ / \
     2 . - . - . - . - . - B - . 2
        \ / \ / \ / \ / \ / \ / \
       3 . - . - . - . - . - . - . 3
          \ / \ / \ / \ / \ / \ / \
         4 . - . - . - . - . - . - . 4
            \ / \ / \ / \ / \ / \ / \
           5 . - . - . - . - . - . - . 5
              \ / \ / \ / \ / \ / \ / \
             6 . - . - . - . - . - . - . 6
                0   1   2   3   4   5   6

```

### Command Line Options
The program supports several command-line options for automated testing and benchmarking:

```bash
# Human vs Dummy AI
./hex_game -m DUMMY

# Dummy vs Brute Force
./hex_game -m "(DUMMY, BRUTE_FORCE)"

# Brute Force vs Monte Carlo
./hex_game -m "(BRUTE_FORCE, MC)"

# Monte-Carlo vs Monte-Carlo Tree Search
# with selection strategy for MCTS set to WIN_RATE
./hex_game -m "(MC, MCTS{strategy:WIN_RATE})"

# 5 games with different AIs: (MC and MCTS)
# with 4 threads for MC and 7 threads for MCTS
./hex_game -g 5 -t 4 -m "(MCTS{threads:7}, MONTE_CARLO)"

# Human vs Monte-Carlo AI on 11x11 board, 4 threads
./hex -s 11 -m \"MC{threads:4}

# Benchmark: 100 AI vs AI games on 7x7 board
./hex -s 7 -g 100 -q -m "(MCTS, MONTE_CARLO)"      
```

Available options:
- `-s, --size SIZE` - Board size (default: 7)
- `-t, --threads THREADS` - Default threads for AI (default: hardware_concurrency)
- `-g, --games GAMES` - Number of games to play (default: 1)
- `-q, --quiet` - Quiet mode - disable most output (statistics only)
- `-m, --machines CONFIG` - Machine configuration (default: HUMAN,HUMAN)
   Examples:
   - MC                    (Human vs Monte Carlo AI)
   - MCTS                  (Human vs MCTS AI)
   - DUMMY                 (Human vs Dummy AI)
   - BRUTE_FORCE           (Human vs Brute Force AI)
   - (MC, MCTS)            (MC AI vs MCTS AI)
   - (MCTS{strategy:WIN_RATE}, MC{threads:4})
- `-h, --help` - Show help message

In quiet mode, the program collects detailed statistics about engine performance including win rates, move times, and efficiency metrics, which are exported to CSV for analysis.

Below is a screen record of a party on a 9x9 board with the machine playing against itself with the 2 engines being objects of type `HexMachineMcIA`.  
![video](./Video/screen_record_hex9.gif)

## Code Explanations
The code for the colored graph I use in this project can be found in my repository [Undirected-Graph-Algorithms](https://github.com/CL-BZH/Undirected-Graph-Algorithms).  

The file `hex.h` contains the class `Hex` that is the entry point for running the Hex game.  
As can be seen in `main.cpp`, to instantiate an `Hex` object one must provide an `HexBoard` object and an `HexMachineEngine` object *(see explanations below)*.  
There can be 0, 1 or 2 human players.  
The method `prompt_player()` is called when it is the turn of the *(blue or red)* human player to play.
The method `machine_play()` is called when it is the turn of the *(blue or red)* machine player to play.

The `HexBoard` class is defined in the file `hex_board.h` and the `HexMachineEngine` class is defined in the file `hex_machine_engine.h`.  
The `HexBoard` object is in charge of the board for the Hex game.
Some of the main methods are:
- `select()`
- `draw_board()`
- `has_won()`

*(there are some more helper functions)*  

`select()` is in charge of the reservation of a cell on the board. It will prompt the player to select a position until she/he selects an empty cell on the board.  
`draw_board()` is in charge of drawing the board on the terminal.  
`has_won()` controls if the current player built a path from one side to the other. It uses the Dijkstra shortest path algorithm implemented in `shortest_path.h`. The algorithm is ran on a colored graph that represents the board.  
> Explanation on why I use a shortest path algorithm to find a winner will be given below.  

The `HexMachineEngine` class is the interface for machine engines.  
All machine engines inherit from `HexMachineEngine` and they define the algorithms that the **machine player** (i.e. the computer) uses to play the game. Currently there are four engines:

### 1. HexMachineDummy
`HexMachineDummy` implements the 'dummy' machine where positions on the board are randomly chosen. This serves as a baseline for comparison with more sophisticated engines.

### 2. HexMachineBF (Brute-Force)
`HexMachineBF` is the 'Brute-Force' engine. This one should be used only on small boards or once there are only few available cells on the board.  
*(e.g. On a 8 cores' computer selecting the first position of a 5x5 board would take about 12 minutes, and about 5 minutes for the second position)*  

> **How it works**:  
The 'Brute-Force' engine generates all the possible states of a given board.  
That is, if there are `n` available cells on the board then all combinations of `k` cells to be given to the current player will be generated. With  
> 
> $$k = \left\lfloor\frac{n+1}{2}\right\rfloor$$
>
> For each combination, the cells that were not selected are given to the opponent.  
Then each cell is given a `quality` score that is the sum of `quality` scores obtained from each combination.  
**For one combination**, the `quality` is set to 0 if there is no path connecting the 2 edges otherwise it is equal to the sum of the inverse of the length of shortest path(s) that connect two opposite edges *(see `has_won()` in `hex_board.h` )*.  

The combinations are generated in parallel across multiple threads using a recursive `n_choose_k()` function that efficiently distributes the workload.

### 3. HexMachineMcIA (Monte Carlo)
`HexMachineMcIA` uses Monte-Carlo simulation to decide which move is the best for a given state of the board.  

> **How it works**  
For each available position:
> * Select that position temporarily
> * If this position creates a winning path, return it immediately
> * Otherwise, spawn threads to run Monte Carlo simulations:
>   - Randomly fill the remaining board positions
>   - Compute quality scores based on shortest path lengths
>   - Calculate average quality for current player and opponent
> * Compute winning probability: `player_quality / (player_quality + opponent_quality)`
> * Select position with highest winning probability

The number of Monte Carlo runs is dynamically adjusted using a bell curve based on board size and available cells, with more runs during critical mid-game positions.

> **Note**:  
There is an option to switch to **brute-force** once the number of free positions is below a certain limit (`_BF_SUP`).

### 4. HexMachineMCTS (Monte Carlo Tree Search)
`HexMachineMCTS` implements a more advanced Monte Carlo Tree Search algorithm that builds a search tree incrementally.

> **How it works**:
> 1. **Selection**: Traverse the tree using UCB1 formula (with exploration parameter C=2.6 by default) to balance exploration vs exploitation
> 2. **Expansion**: Add new nodes to the tree for unexplored moves
> 3. **Simulation**: Run random playouts from newly expanded nodes
> 4. **Backpropagation**: Update node statistics with playout results

Key features:
- **Root parallelization**: Multiple threads build separate trees that are merged
- **Atomic operations**: Thread-safe node updates using `std::atomic` and custom `AtomicDouble`
- **Dynamic iterations**: Number of iterations adapts based on game phase
- **Standard rewards**: Uses binary win/loss outcomes
- **UCB1 exploration constant adapted to the board size**:

   The UCB1 formula is $ucb = \frac{w}{n} + c \times \sqrt{\ln\frac{N}{n}}$.

   Therefore:
   - Larger boards → more moves → larger N → exploration term grows naturally
   - Smaller boards → fewer moves → smaller N → need higher c to encourage exploration

   Why c depends on board size:

      Branching factor: Larger boards have more possible moves
      Game complexity: More positions to explore
      Depth of game: Longer games require different exploration/exploitation balance
   
   The value of c is computed in the MCTS engine by the method get_exploration_constant.

   For example, for a board of size 5, 7, 9 or 11
   the function returns 2.64, 2.28, 1.92 and 1.56 respectively.

Based on current testing results, the flat Monte Carlo engine demonstrates stronger performance with higher win rates compared to the MCTS implementation, despite MCTS's more sophisticated tree search methodology.

## Statistics and Analysis
The framework includes comprehensive statistics collection (`hex_statistics.h`) that tracks:
- Win rates for each engine and starting position
- Average move times
- Number of moves per game
- Efficiency metrics (win rate per second of thinking time)

In quiet mode (`-q` flag), statistics are exported to CSV format for further analysis. A Jupyter notebook is provided for visualizing and comparing engine performance.

## Performance Characteristics
- **Dummy**: Fastest, random moves
- **Brute-Force**: Optimal but exponentially slow, only practical for very small boards
- **Monte Carlo**: Good balance of speed and quality, parallelizable
- **MCTS**: More sophisticated search but currently shows lower win rates than Monte Carlo; may benefit from further tuning of exploration parameters and playout strategies

----
## To Do
- Build a Reinforcement Learning solution...
- Optimize MCTS with neural network guidance
- Add opening book support
- Implement transposition tables for graph isomorphism
```

