    The file "connect4.py" is a Python script that simulates the game "Connect 4" and implements the adversarial search algorithms Minimax, Minimax with Alpha-Beta pruning, and Monte Carlo Tree Search (exceptionally, we also included the "Dummy" algorithm, which makes random moves).

    Interaction with the game is done through the terminal. After each move, the current state of the board and instructions for each player are printed. The game can be played with 2 human players, one human player and the other using one of the implemented algorithms (in this case, the human player is always player 1), or an algorithm against another algorithm.

    To run the script, make sure you are in the correct directory and have Python interpreter and any necessary libraries installed on your machine. Then, execute the following command in the command line: python connect4.py <game_type> <depth/number of iterations>. This command runs the game with the chosen parameters specified as command line arguments.

To select the game type, choose one of the algorithms (they must be written exactly in the following way, this will be the first argument):
- Human: 2 human players.
- Minimax: Player 1 is human, and Player 2 is the Minimax algorithm without Alpha-Beta pruning.
- MinimaxAlphaBeta: Player 1 is human, and Player 2 is the Minimax algorithm with Alpha-Beta pruning.
- MCTS: Player 1 is human, and Player 2 is the Monte Carlo Tree Search algorithm.
- Dummy: Player 1 is human, and Player 2 is the Random algorithm.
Next, choose the depth (recommended: 3 for Minimax, 5 for Minimax with Alpha-Beta pruning, and 20000 for MCTS).

For example, "python connect4.py Minimax 3" initializes the game where player 2's moves are decided by the Minimax algorithm with a depth of 3.
