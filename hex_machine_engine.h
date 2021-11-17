#ifndef HEX_MACHINE_ENGINE_H
#define HEX_MACHINE_ENGINE_H

struct HexMachineEngine {

HexMachineEngine(unsigned int board_size): gen(rd()),
    uniform_distribution(std::uniform_int_distribution<int>(0, board_size-1)) {}
  
  virtual void get_position(unsigned int& row, unsigned int& column) = 0;
  
protected:
  
  // obtain a random number from hardware
  std::random_device rd;
  
  // generator
  std::mt19937 gen;
  
  // Uniform distibution over the range [0, board_size - 1]
  // Note: std::uniform_int_distribution<> is inclusive
  std::uniform_int_distribution<int> uniform_distribution;
  
};

// Dummy machine.
// Just return random value for the row and the column in
// range [0, board_size - 1]
// It is the caller responsability to check the validity of the position
// (i.e. the position (row, column) is available on the board)
struct HexMachineDummy: HexMachineEngine {
  
 HexMachineDummy(unsigned int board_size): HexMachineEngine(board_size) {}
  
  void get_position(unsigned int& row, unsigned int& column) override {
    // Stupid machine:
    // generate random numbers for the row and the column
    row = uniform_distribution(gen); 
    column = uniform_distribution(gen); 
  }
};

// Monte-Carlo IA
// Not yet implemented (just do the same as the dummy machine for the moment)
struct HexMachineMcIA: HexMachineEngine {
  
 HexMachineMcIA(unsigned int board_size): HexMachineEngine(board_size) {}
  
  void get_position(unsigned int& row, unsigned int& column) override {
    // Stupid machine:
    // generate random numbers for the row and the column
    row = uniform_distribution(gen); 
    column = uniform_distribution(gen); 
  }
  
};

  
#endif //def HEX_MACHINE_ENGINE_H
