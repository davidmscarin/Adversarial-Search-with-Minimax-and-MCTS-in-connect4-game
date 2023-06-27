import numpy as np
import sys
import random
import math
import time
import matplotlib.pyplot as plt


ROWS, COLS = 6, 7
initial_board = np.zeros((ROWS, COLS))


class Node:
    def __init__(self, board, player, move=None):
        self.board = board
        self.player = player
        self.move = move
        self.score = 0
        self.visits = 0
        self.children = []
        self.parent = None
    
    def is_fully_expanded(self):
        return len(self.children) == 7
    
    def select_child(self, exploration_parameter):
        scores = []
        for child in self.children:
            exploitation_score = child.wins / child.visits
            exploration_score = exploration_parameter * math.sqrt(math.log(self.visits) / child.visits)
            scores.append(exploitation_score + exploration_score)
        return self.children[scores.index(max(scores))]
    
    def expand(self):
        available_moves = self.board.get_valid_moves()
        for move in available_moves:
            new_board = self.board.copy()
            new_board.play_move(move, self.player)
            self.children.append(Node(new_board, switch_turn(self.player)))
    
    def update(self, winner):
        self.visits += 1
        if winner == self.player:
            self.wins += 1


class Board:
    def __init__(self, state, spaces = 42):
        self.state = state
        self.spaces = spaces
        
    def col_complete(self, a):
        return self.state[0][a]!=0
    
    def check_spaces(self):
        if self.spaces==0:
            print("Game Draw")
            exit()

    def play_move(self, a, Player):
        if a > 6 or a < 0:
            return False
        
        if Player == 1:
            if not self.col_complete(a):
                for i in range(ROWS-1, -1, -1):
                    if self.state[i][a] == 0:
                        self.state[i][a] = Player
                        break
        else:
            if not self.col_complete(a):
                for i in range(ROWS-1, -1, -1):
                    if self.state[i][a] == 0:
                        self.state[i][a] = Player
                        break
        
        self.spaces-=1
        return True

    def four_in_line(self):

        blu_pos = []
        red_pos = []

        for i in range(ROWS):
            for j in range(COLS):
                if self.state[i][j]==1:
                    blu_pos.append((i, j))
                elif self.state[i][j]==2:
                    red_pos.append((i, j))

        for pos in blu_pos:
            a, b = pos
            try:
                if (self.state[a+1, b+1]==1 and self.state[a+2, b+2]==1 and self.state[a+3, b+3]==1) or (self.state[a-1, b-1]==1 and self.state[a-2, b-2]==1 and self.state[a-3, b-3]==1):
                    return 1
            except IndexError:
                pass

            try:
                if (self.state[a, b+1]==1 and self.state[a, b+2]==1 and self.state[a, b+3]==1) or (self.state[a, b-1]==1 and self.state[a, b-2]==1 and self.state[a, b-3]==1):
                    return 1
            except IndexError:
                pass

            try:
                if (self.state[a+1, b]==1 and self.state[a+2, b]==1 and self.state[a+3, b]==1) or (self.state[a-1, b]==1 and self.state[a-2, b]==1 and self.state[a-3, b]==1):
                    return 1
            except IndexError:
                pass

        for pos in red_pos:
            a, b = pos
            try:
                if (self.state[a+1, b+1]==2 and self.state[a+2, b+2]==2 and self.state[a+3, b+3]==2) or (self.state[a-1, b-1]==2 and self.state[a-2, b-2]==2 and self.state[a-3, b-3]==2):
                    return 2
            except IndexError:
                pass

            try:
                if (self.state[a, b+1]==2 and self.state[a, b+2]==2 and self.state[a, b+3]==2) or (self.state[a, b-1]==2 and self.state[a, b-2]==2 and self.state[a, b-3]==2):
                    return 2
            except IndexError:
                pass

            try:
                if (self.state[a+1, b]==2 and self.state[a+2, b]==2 and self.state[a+3, b]==2) or (self.state[a-1, b]==2 and self.state[a-2, b]==2 and self.state[a-3, b]==2):
                    return 2
            except IndexError:
                pass

        return 0
    
    def is_game_over(self):
        n = self.four_in_line()
        if n == 1:
            return 1
        if n == 2:
            return 2

        return 0

    def eval_diag(self, i, j, max_player, min_player):

        max_count = 0
        min_count = 0
        max_player_count = 0
        min_player_count = 0

        for i in range(i, i+3):
            for j in range(j, j+3):
                if i==j:
                    if self.state[i][j] == max_player:
                        max_player_count+=1
                    elif self.state[i][j] == min_player:
                        min_player_count+=1

        max_count+=max_player_count
        min_count+=min_player_count

        max_player_count=0
        min_player_count=0

        for i in range (ROWS):
            for j in range(COLS-1, COLS-3, -1):
                if j == COLS-i:
                    if self.state[i][j] == max_player:
                        max_player_count+=1
                    elif self.state[i][j] == min_player:
                        min_player_count+=1
        
        max_count+=max_player_count
        min_count+=min_player_count

        if max_count == 0 and min_count == 3:
            return -50

        if max_count == 0 and min_count == 2:
            return -20

        if max_count == 0 and min_count == 1:
            return -1

        if max_count == 1 and min_count == 0:
            return 1

        if max_count == 2 and min_player_count == 0:
            return 10

        if max_count == 3 and min_count == 0:
            return 50

        return 0
    
    def eval_square_rows(self, i, j, max_player, min_player):

        max_player_count = 0
        min_player_count = 0

        for i in range(i, i + 3):
            if (self.state[i][j] == max_player):
                max_player_count += 1
            elif (self.state[i][j] == min_player):
                min_player_count += 1

        if max_player_count == 0 and min_player_count == 3:
            return -50

        if max_player_count == 0 and min_player_count == 2:
            return -20

        if max_player_count == 0 and min_player_count == 1:
            return -1

        if max_player_count == 1 and min_player_count == 0:
            return 1

        if max_player_count == 2 and min_player_count == 0:
            return 10

        if max_player_count == 3 and min_player_count == 0:
            return 50

        return 0

    def eval_square_cols(self, i, j, max_player, min_player):

        max_player_count = 0
        min_player_count = 0

        for i in range(j, j + 3):
            if (self.state[i][j] == max_player):
                max_player_count += 1
            elif (self.state[i][j] == min_player):
                min_player_count += 1

        if max_player_count == 4:
            return 512

        if min_player_count:
            return -512

        if max_player_count == 0 and min_player_count == 3:
            return -50

        if max_player_count == 0 and min_player_count == 2:
            return -20

        if max_player_count == 0 and min_player_count == 1:
            return -1

        if max_player_count == 1 and min_player_count == 0:
            return 1

        if max_player_count == 2 and min_player_count == 0:
            return 10

        if max_player_count == 3 and min_player_count == 0:
            return 50

        return 0

    def evaluation(self):

        evaluation = 0

        for i in range(ROWS):
            for j in range(COLS):
                if (i + 3) < ROWS:
                    evaluation += self.eval_square_rows(i, j, 1, 2)
                if (j + 3) < COLS:
                    evaluation += self.eval_square_cols(i, j, 1, 2)
                if (i+3) < ROWS and (j+3) < COLS:
                    evaluation += self.eval_diag(i, j, 1, 2)

        return evaluation
    
    
    def get_possible_moves(self):

        possible_moves = []

        for j in range(COLS):
            for i in range(ROWS-1, -1, -1):
                if(self.state[i][j]==0):
                    possible_moves.append((i, j))
                    break
        
        return possible_moves
    
    def get_possible_states(self, player):
        
        possible_moves = self.get_possible_moves()
        possible_states = []

        for move in possible_moves:
            possible_state = np.copy(self.state)
            i, j = move
            if player==1:
                possible_state[i][j] = 1
            else:
                possible_state[i][j] = 2
            possible_states.append((possible_state, move))
        
        return possible_states

    def minimax(self, depth, maximizing_player, print_info=True):
        if depth == 0 or self.is_game_over()!=0:
            return self.evaluation(), None
        if maximizing_player:
            value = -float("inf")
            best_move = None
            possible_states = self.get_possible_states(1)
            for state, move in possible_states:
                possible_board = Board(state)
                new_value, possible_move = possible_board.minimax(depth-1, False, False)
                if new_value > value:
                    value = new_value
                    best_move = move
            if print_info:
                print("Maximizing player found move %s with value %s" % (best_move, value))
            return value, best_move
        else:
            value = float("inf")
            best_move = None
            possible_states = self.get_possible_states(2)
            for state, move in possible_states:
                possible_board = Board(state)
                new_value, possible_move = possible_board.minimax(depth-1, True, False)
                if new_value < value:
                    value = new_value
                    best_move = move
            if print_info:
                print("Minimizing player found move %s with value %s" % (best_move, value))
            return value, best_move

        
    def minimax_helper(self, depth, maximizing_player):
        start_time = time.time()
        value, best_move = self.minimax(depth, maximizing_player)
        end_time = time.time()
        print("Time to find a move: %.2f seconds" % (end_time - start_time))
        runtime = round(abs(start_time-end_time), 2)
        return runtime
    
        
    def minimax_alpha_beta(self, depth, alpha, beta, maximizing_player, print_info=False):
        if depth == 0 or self.is_game_over()!=0:
            return self.evaluation(), None, 1
        if maximizing_player:
            value = -float("inf")
            best_move = None
            nodes_pruned = 0
            possible_states = self.get_possible_states(1)
            for next_state, move in possible_states:
                possible_board = Board(next_state)
                new_value, possible_move, pruned = possible_board.minimax_alpha_beta(depth-1, alpha, beta, False, False)
                if new_value > value:
                    value = new_value
                    best_move = move  
                alpha = max(alpha, value)
                if alpha >= beta:
                    nodes_pruned += 1
                    break
                nodes_pruned += pruned
            if print_info:
                print("Maximizing player found move %s with value %s" % (best_move, value))
            return value, best_move, nodes_pruned
        else:
            value = float("inf")
            best_move = None
            nodes_pruned = 0
            possible_states = self.get_possible_states(2)
            for next_state, move in possible_states:
                possible_board = Board(next_state)
                new_value, possible_move, pruned = possible_board.minimax_alpha_beta(depth-1, alpha, beta, True, False)
                if new_value < value:
                    value = new_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    nodes_pruned += 1
                    break
                nodes_pruned += pruned
            if print_info:
                print("Minimizing player found move %s with value %s" % (best_move, value))
            return value, best_move, nodes_pruned


    def minimax_alpha_beta_helper(self, depth, alpha, beta, maximizing_player, print_info=False):
        start_time = time.time()
        value, best_move, nodes_pruned = self.minimax_alpha_beta(depth, alpha, beta, maximizing_player)
        end_time = time.time()
        if print_info==True:
            print("Time to find a move: %.2f seconds" % (end_time - start_time))
            print("Nodes pruned: %s" % nodes_pruned)
        runtime = round(abs(start_time-end_time), 2)
        return runtime
    
    def mcts(self, max_iterations):
        root = Node(self.state, 2)
        state_copy = np.copy(self.state)
        board_copy = Board(state_copy) 

        for i in range(max_iterations):
            node = root
            board = Board(board_copy.state)
            player = node.player


            while node.children:
                node = max(node.children, key=lambda child: child.score / child.visits + math.sqrt(2 * math.log(1+node.visits) / max(1, child.visits)))
                board.play_move(node.move, node.player)
                player = 3 - player 


            possible_moves = board.get_possible_moves()
            if possible_moves:
                move = random.choice(possible_moves)
                row, col = move
                board.play_move(col, player)
                node.children.append(Node(board.state, player, col))
                node = node.children[-1]


            while not board.is_game_over():
                move = random.choice(board.get_possible_moves())
                row, col = move
                board.play_move(col, player)
                player = 3 - player  


            while node:
                node.visits += 1
                if node.player == player:
                    node.score += 1
                node = node.parent

        return max(root.children, key=lambda child: child.visits).move
    
    def mcts_helper(self, iterations):
        start_time = time.time()
        best_move = self.mcts(iterations)
        end_time = time.time()
        runtime = abs(end_time-start_time)
        return runtime

    def dummy(self):
        possible_states = self.get_possible_moves()
        random_move = random.choice(possible_states)
        row, col = random_move
        return col

def switch_turn(Player):
    return 3 - Player


state = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 2, 2, 0, 0, 0], [0, 0, 1, 2, 0, 0, 0], [0, 0, 1, 1, 2, 0, 0], [0, 0, 1, 2, 2, 1, 0]]
state = np.array(state)

times = []
depths = []

times_alpha = []
depths_alpha = []

times_mcts = []
iterations = []

board = Board(state)

for i in range(2, 8):
    runtime = board.minimax_helper(i, False)
    times.append(runtime)
    depths.append(i)



for i in range(2, 10):
    runtime = board.minimax_alpha_beta_helper(i, -float("inf"), float("inf"), False)
    times_alpha.append(runtime)
    depths_alpha.append(i)


for i in range(10000, 30000, 2500):
    runtime = board.mcts_helper(i)
    times_mcts.append(runtime)
    iterations.append(i)

'MINIMAX'

plt.scatter(depths, times, color='red', marker='o')

# Add labels and a title
plt.xlabel('Depth')
plt.ylabel('Time')
plt.title('Time to find a move by depth Minimax')

plt.show()

'''ALPHA BETA'''

plt.scatter(depths_alpha, times_alpha, color='blue', marker='o')

# Add labels and a title
plt.xlabel('Depth')
plt.ylabel('Time')
plt.title('Time to find a move by depth Alpha Beta')

plt.show()


'''COMPARISON'''

plt.scatter(depths, times, color='red', marker='o')
plt.scatter(depths_alpha, times_alpha, color='blue', marker='o')

plt.xlabel('Depth')
plt.ylabel('Time')
plt.title('Comparison with alpha beta pruning')

plt.show()


'''MCTS'''

plt.scatter(iterations, times_mcts, color='red', marker='o')

plt.xlabel('Iterations')
plt.ylabel('Time')
plt.title('MCTS time by no. iterations')

plt.show()