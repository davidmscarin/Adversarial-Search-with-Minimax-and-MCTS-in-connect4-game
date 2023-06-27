import numpy as np
import sys
import random
import math
import time
import pygame



#inicializacao
ROWS, COLS = 6, 7
initial_board = np.zeros((ROWS, COLS))
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BOARD_ROWS = 6
BOARD_COLS = 7
SQUARE_SIZE = 60
GRID_WIDTH = SQUARE_SIZE * BOARD_COLS
GRID_HEIGHT = SQUARE_SIZE * BOARD_ROWS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
font = pygame.font.SysFont(None, 36)
pygame.display.set_caption('Connect4')

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
    
    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def select_child(self, exploration_parameter):
        scores = []
        for child in self.children:
            exploitation_score = child.wins / child.visits
            exploration_score = exploration_parameter * math.sqrt(math.log(self.visits) / child.visits)
            scores.append(exploitation_score + exploration_score)
        return self.children[scores.index(max(scores))]
    
    def get_ucb_score(self, exploration_parameter):
        exploit = self.wins / self.visits
        explore = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploit + exploration_parameter * explore

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
    def __init__(self, state, spaces=42):
        self.state = state
        self.spaces = spaces

    def col_complete(self, a):
        return self.state[0][a] != 0

    def check_spaces(self):
        if self.spaces == 0:
            return 0

    def play_move(self, a, Player):
        if a > 6 or a < 0:
            return False

        if Player == 1:
            if not self.col_complete(a):
                for i in range(ROWS - 1, -1, -1):
                    if self.state[i][a] == 0:
                        self.state[i][a] = Player
                        break
        else:
            if not self.col_complete(a):
                for i in range(ROWS - 1, -1, -1):
                    if self.state[i][a] == 0:
                        self.state[i][a] = Player
                        break

        self.spaces -= 1
        return True

    def four_in_line(self):
        up_right = [(1, 1), (2, 2), (3, 3)]
        up_left = [(1, -1), (2, -2), (3, -3)]
        down_right = [(-1, 1), (-2, 2), (-3, 3)]
        down_left = [(-1, -1), (-2, -2), (-3, -3)]
        right = [(0, 1), (0, 2), (0, 3)]
        left = [(0, -1), (0, -2), (0, -3)]
        up = [(1, 0), (2, 0), (3, 0)]
        down = [(-1, 0), (-2, 0), (-3, 0)]
        list_of_vectors = [up_right, up_left, down_right, down_left, right, left, up, down]

        blu_pos = []
        red_pos = []

        for i in range(ROWS):
            for j in range(COLS):
                if self.state[i][j] == 1:
                    blu_pos.append((i, j))
                elif self.state[i][j] == 2:
                    red_pos.append((i, j))

        for pos in blu_pos:
            c, d = pos
            for vectors in list_of_vectors:
                counter = 0
                for vector in vectors:
                    a, b = vector
                    try:
                        if (self.state[c + a][d + b] != 1) or (c + a) < 0 or (d + b) < 0:
                            counter = 0
                        else:
                            counter += 1
                    except:
                        pass
                if counter == 3:
                    return 1

        for pos in red_pos:
            c, d = pos
            for vectors in list_of_vectors:
                counter = 0
                for vector in vectors:
                    a, b = vector
                    try:
                        if (self.state[c + a][d + b] != 2) or (c + a) < 0 or (d + b) < 0:
                            counter = 0
                        else:
                            counter += 1
                    except:
                        pass
                if counter == 3:
                    return 2

        return 0

    def is_game_over(self):
        if self.check_spaces()==0:
            return -1
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

    def visual(self):
        print("\n=============================")
        for i in range(6):
            for j in range(7):
                print('| {} '.format('X' if self.state[i][j] == 1 else 'O' if self.state[i][j] == 2 else ' '),
                      end='')
            print("|")

        print("  1   2   3   4   5   6   7")
        print('=============================')

    def get_possible_moves(self):

        possible_moves = []

        for j in range(COLS):
            for i in range(ROWS - 1, -1, -1):
                if (self.state[i][j] == 0):
                    possible_moves.append((i, j))
                    break

        return possible_moves

    def get_possible_states(self, player):

        possible_moves = self.get_possible_moves()
        possible_states = []

        for move in possible_moves:
            possible_state = np.copy(self.state)
            i, j = move
            if player == 1:
                possible_state[i][j] = 1
            else:
                possible_state[i][j] = 2
            possible_states.append((possible_state, move))

        return possible_states

    def minimax(self, depth, maximizing_player, print_info=True):
        if depth == 0 or self.is_game_over() != 0:
            return self.evaluation(), None
        if maximizing_player:
            value = -float("inf")
            best_move = None
            possible_states = self.get_possible_states(1)
            for state, move in possible_states:
                possible_board = Board(state)
                new_value, possible_move = possible_board.minimax(depth - 1, False, False)
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
                new_value, possible_move = possible_board.minimax(depth - 1, True, False)
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
        return best_move

    def minimax_alpha_beta(self, depth, alpha, beta, maximizing_player, print_info=True):
        if depth == 0 or self.is_game_over() != 0:
            return self.evaluation(), None, 1
        if maximizing_player:
            value = -float("inf")
            best_move = None
            nodes_pruned = 0
            possible_states = self.get_possible_states(1)
            for next_state, move in possible_states:
                possible_board = Board(next_state)
                new_value, possible_move, pruned = possible_board.minimax_alpha_beta(depth - 1, alpha, beta, False,
                                                                                     False)
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
                new_value, possible_move, pruned = possible_board.minimax_alpha_beta(depth - 1, alpha, beta, True,
                                                                                     False)
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
        

    def minimax_alpha_beta_helper(self, depth, alpha, beta, maximizing_player):
        start_time = time.time()
        value, best_move, nodes_pruned = self.minimax_alpha_beta(depth, alpha, beta, maximizing_player)
        end_time = time.time()
        print("Time to find a move: %.2f seconds" % (end_time - start_time))
        print("Nodes pruned: %s" % nodes_pruned)
        return best_move
    

    def mcts(self, max_iterations):
        root = Node(self.state, 2)
        state_copy = np.copy(self.state)
        board_copy = Board(state_copy)

        for i in range(max_iterations):
            node = root
            board = Board(board_copy.state)
            player = node.player

            while node.children:
                node = max(node.children, key=lambda child: child.score / child.visits + math.sqrt(
                    2 * math.log(1 + node.visits) / max(1, child.visits)))
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


    def dummy(self):
        possible_states = self.get_possible_moves()
        random_move = random.choice(possible_states)
        row, col = random_move
        return col

    def draw_in_board(self):
        for i in range(ROWS):
            for j in range(COLS):
                if(self.state[i][j]==1):
                    draw_circle(BLUE, i, j)
                elif self.state[i][j]==2:
                    draw_circle(RED, i, j)


def switch_turn(Player):
    return 3 - Player

def draw_circle(color, row, col, offset_x = (SCREEN_WIDTH - GRID_WIDTH) // 2, offset_y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2):
    circle_x = offset_x + (col * SQUARE_SIZE) + (SQUARE_SIZE // 2)
    circle_y = offset_y + (row * SQUARE_SIZE) + (SQUARE_SIZE // 2)
    pygame.draw.circle(screen, color, (circle_x, circle_y), SQUARE_SIZE // 2)


''' GAME START '''

prompt = sys.argv[1] #algoritmo a executar

if(prompt!="Dummy" and prompt!="Human"):
    prompt2 = int(sys.argv[2]) #parametros alterados no algoritmo (nomeadamente profundidade)

if prompt == "Human":
    board = Board(initial_board)
    Player = 1
    print(board.state)
    screen.fill(WHITE)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONUP:

                mouse_x, mouse_y = pygame.mouse.get_pos()
    
                row = (mouse_y - offset_y) // SQUARE_SIZE
                col = (mouse_x - offset_x) // SQUARE_SIZE
                board.play_move(col, Player)
                print(board.state)
                board.draw_in_board()
                game_over_var = board.is_game_over()
                if game_over_var==-1:
                    print("Draw")
                    winning_text = font.render(f"Draw!", True, BLUE)
                    screen.blit(winning_text, (offset_x, offset_y - 40))
                    pygame.display.flip()
                    time.sleep(2)
                    sys.exit()
                elif game_over_var!=0:
                    print(f'Player {Player} wins!')
                    winning_text = font.render(f"Player {Player} Wins!", True, BLUE)
                    screen.blit(winning_text, (offset_x, offset_y - 40))
                    pygame.display.flip()
                    time.sleep(2)
                    sys.exit()
                
                Player = switch_turn(Player)
        
        offset_x = (SCREEN_WIDTH - GRID_WIDTH) // 2
        offset_y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

        for row in range(BOARD_ROWS + 1):
            pygame.draw.line(screen, BLACK, (offset_x, offset_y + row * SQUARE_SIZE), (offset_x + GRID_WIDTH, offset_y + row * SQUARE_SIZE))
        for col in range(BOARD_COLS + 1):
            pygame.draw.line(screen, BLACK, (offset_x + col * SQUARE_SIZE, offset_y), (offset_x + col * SQUARE_SIZE, offset_y + GRID_HEIGHT))


        pygame.display.flip()


elif prompt == "Minimax":

    board = Board(initial_board)
    Player = 1
    print(board.state)
    screen.fill(WHITE)
    while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONUP and Player==1:
            
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    row = (mouse_y - offset_y) // SQUARE_SIZE
                    col = (mouse_x - offset_x) // SQUARE_SIZE
                    board.play_move(col, Player)
                    print(board.state)
                    board.draw_in_board()
                    board.check_spaces()
                    game_over_var = board.is_game_over()

                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()

                    Player = switch_turn(Player)
                    pygame.display.flip()
                
                if Player==2:
                    
                    move = board.minimax_helper(prompt2, False)
                    row, col = move
                    board.check_spaces()
                    board.play_move(col, Player)
                    print(board.state)
                    board.draw_in_board()
                    game_over_var = board.is_game_over()

                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    
                    Player = switch_turn(Player)
                    pygame.display.flip()

            
            offset_x = (SCREEN_WIDTH - GRID_WIDTH) // 2
            offset_y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

            for row in range(BOARD_ROWS + 1):
                pygame.draw.line(screen, BLACK, (offset_x, offset_y + row * SQUARE_SIZE), (offset_x + GRID_WIDTH, offset_y + row * SQUARE_SIZE))
            for col in range(BOARD_COLS + 1):
                pygame.draw.line(screen, BLACK, (offset_x + col * SQUARE_SIZE, offset_y), (offset_x + col * SQUARE_SIZE, offset_y + GRID_HEIGHT))

            pygame.display.flip()


elif prompt == "MinimaxAlphaBeta":
    
    board = Board(initial_board)
    Player = 1
    print(board.state)
    screen.fill(WHITE)
    while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONUP and Player==1:
            
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    row = (mouse_y - offset_y) // SQUARE_SIZE
                    col = (mouse_x - offset_x) // SQUARE_SIZE
                    board.play_move(col, Player)
                    print(board.state)
                    board.draw_in_board()
                    board.check_spaces()
                    game_over_var = board.is_game_over()

                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()

                    Player = switch_turn(Player)
                    pygame.display.flip()
                
                if Player==2:
                    
                    move = board.minimax_alpha_beta_helper(prompt2, -float("inf"), float("inf"), False)
                    row, col = move
                    board.check_spaces()
                    board.play_move(col, Player)
                    print(board.state)
                    board.draw_in_board()
                    game_over_var = board.is_game_over()

                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    
                    Player = switch_turn(Player)
                    pygame.display.flip()

            
            offset_x = (SCREEN_WIDTH - GRID_WIDTH) // 2
            offset_y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

            for row in range(BOARD_ROWS + 1):
                pygame.draw.line(screen, BLACK, (offset_x, offset_y + row * SQUARE_SIZE), (offset_x + GRID_WIDTH, offset_y + row * SQUARE_SIZE))
            for col in range(BOARD_COLS + 1):
                pygame.draw.line(screen, BLACK, (offset_x + col * SQUARE_SIZE, offset_y), (offset_x + col * SQUARE_SIZE, offset_y + GRID_HEIGHT))

            pygame.display.flip()

elif prompt == 'MCTS':
   
    board = Board(initial_board)
    Player = 1
    print(board.state)
    screen.fill(WHITE)
    while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONUP and Player==1:
            
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    row = (mouse_y - offset_y) // SQUARE_SIZE
                    col = (mouse_x - offset_x) // SQUARE_SIZE
                    board.play_move(col, Player)
                    print(board.state)
                    board.draw_in_board()
                    board.check_spaces()
                    game_over_var = board.is_game_over()

                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()

                    Player = switch_turn(Player)
                    pygame.display.flip()
                
                if Player==2:
                    
                    move = board.mcts(prompt2)
                    board.check_spaces()
                    board.play_move(move, Player)
                    print(board.state)
                    board.draw_in_board()
                    game_over_var = board.is_game_over()

                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    
                    Player = switch_turn(Player)
                    pygame.display.flip()

            
            offset_x = (SCREEN_WIDTH - GRID_WIDTH) // 2
            offset_y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

            for row in range(BOARD_ROWS + 1):
                pygame.draw.line(screen, BLACK, (offset_x, offset_y + row * SQUARE_SIZE), (offset_x + GRID_WIDTH, offset_y + row * SQUARE_SIZE))
            for col in range(BOARD_COLS + 1):
                pygame.draw.line(screen, BLACK, (offset_x + col * SQUARE_SIZE, offset_y), (offset_x + col * SQUARE_SIZE, offset_y + GRID_HEIGHT))

            pygame.display.flip()

elif prompt == 'Dummy':
    
    board = Board(initial_board)
    Player = 1
    print(board.state)
    screen.fill(WHITE)
    while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONUP and Player==1:
            
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    row = (mouse_y - offset_y) // SQUARE_SIZE
                    col = (mouse_x - offset_x) // SQUARE_SIZE
                    if board.play_move(col, Player):
                        Player = switch_turn(Player)
                    print(board.state)
                    board.draw_in_board()
                    board.check_spaces()
                    game_over_var = board.is_game_over()

                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    
                    pygame.display.flip()
                
                if Player==2:
                    
                    move = board.dummy()
                    board.check_spaces()
                    board.play_move(move, Player)
                    print(board.state)
                    board.draw_in_board()
                    game_over_var = board.is_game_over()
                    if game_over_var==-1:
                        print("Draw")
                        winning_text = font.render(f"Draw!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    elif game_over_var!=0:
                        print(f'Player {Player} wins!')
                        winning_text = font.render(f"Player {Player} Wins!", True, BLACK)
                        screen.blit(winning_text, (offset_x, offset_y - 40))
                        pygame.display.flip()
                        time.sleep(2)
                        sys.exit()
                    
                    Player = switch_turn(Player)
                    pygame.display.flip()

            
            offset_x = (SCREEN_WIDTH - GRID_WIDTH) // 2
            offset_y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

            for row in range(BOARD_ROWS + 1):
                pygame.draw.line(screen, BLACK, (offset_x, offset_y + row * SQUARE_SIZE), (offset_x + GRID_WIDTH, offset_y + row * SQUARE_SIZE))
            for col in range(BOARD_COLS + 1):
                pygame.draw.line(screen, BLACK, (offset_x + col * SQUARE_SIZE, offset_y), (offset_x + col * SQUARE_SIZE, offset_y + GRID_HEIGHT))

            pygame.display.flip()
