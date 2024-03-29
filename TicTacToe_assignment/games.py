"""Games or Adversarial Search (Chapter 5)"""

import copy
import itertools
import random
from collections import namedtuple

import numpy as np

#from utils import vector_add

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)


# ______________________________________________________________________________
# MinMax Search


def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


def minmax_cutoff(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func."""
    
    """Replace result (which computes utility) with result_cutoff (with evaluation function)
    so that game.utility(state, player) returns the evaluation stored in utility var"""
    
    player = game.to_move(state)
    depth = game.d

    def max_value(state, depth):
        if depth <= 0:
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result_cutoff(state, a), depth - 1))
        return v

    def min_value(state, depth):
        if depth <= 0:
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result_cutoff(state, a), depth - 1))
        return v

    #print("minmax_cutoff: to be done by students")
    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result_cutoff(state, a), depth))

# ______________________________________________________________________________


def expect_minmax(game, state):
    """
    [Figure 5.11]
    Return the best move for a player after dice are thrown. The game tree
	includes chance nodes along with min and max nodes.
	"""
 
    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(game.result(state, a)))
        return v

    def chance_node(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        sum_chances = 0
        num_chances = len(game.chances(state))
        #print("chance_node: to be completed by students")
        
        # return the average over all possible outcomes of the chance nodes
        # sum_chances += probability(successor) * value(successor)
        index = 0
        for a in game.actions(state):
            
            #p = probability(successor)
            #sum_chances += p * value(successor)
            
            successor_prob = game.chances(state)[index]
            successor_val = max_value(game.result(state, a))
            sum_chances += successor_prob * successor_val
            
            #sum_chances += max_value(game.result(state, a)) * game.chances(state)[index]
            index += 1
        return sum_chances

    # Body of expect_minmax:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)), default=None)

def expect_minmax_cutoff(game, state):
    
    """Replace result (which computes utility) with result_cutoff (with evaluation function)
    so that game.utility(state, player) returns the evaluation stored in utility var"""
    
    player = game.to_move(state)
    depth = game.d

    def max_value(state, depth):
        if depth <= 0:
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(game.result_cutoff(state, a), depth - 1))
        return v

    def min_value(state, depth):
        if depth <= 0:
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(game.result_cutoff(state, a), depth - 1))
        return v

    def chance_node(state, depth):
        if depth <= 0:
            return game.utility(state, state)
        sum_chances = 0
        num_chances = len(game.chances(state))
        #print("chance_node: to be completed by students")
        
        # return the average over all possible outcomes of the chance nodes
        # sum_chances += probability(successor) * value(successor)
        index = 0
        for a in game.actions(state):
            
            #p = probability(successor)
            #sum_chances += p * value(successor)
            
            successor_prob = game.chances(state)[index]
            # using min_value here instead of max_value because all the evaluations are inverted
            successor_val = min_value(game.result_cutoff(state, a), depth - 1)
            sum_chances += successor_prob * successor_val
            
            #sum_chances += max_value(game.result(state, a), depth - 1) * game.chances(state)[index]
            index += 1
        return sum_chances

    # Body of expect_minmax:
    return max(game.actions(state), key=lambda a: min_value(game.result_cutoff(state, a), depth), default=None)


def alpha_beta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""
    # Very similar to minimax but with alpha beta pruning

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        #print("alpha_beta_search: max_value: to be completed by student")
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        #print("alpha_beta_search: min_value: to be completed by student")
        return v

    # Body of alpha_beta_search:
    #print("alpha_beta_search: to be completed by students")
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), -np.inf, np.inf))


def alpha_beta_cutoff_search(game, state, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    
    """Replace result (which computes utility) with result_cutoff (with evaluation function)
    so that game.utility(state, player) returns the evaluation stored in utility var"""
    
    #print("alpha_beta_cutoff_search: may be used, if so, must be implemented by students")
    
    player = game.to_move(state)
    depth = game.d

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if depth <= 0:
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result_cutoff(state, a), alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        #print("alpha_beta_search: max_value: to be completed by student")
        return v

    def min_value(state, alpha, beta, depth):
        if depth <= 0:
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result_cutoff(state, a), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        #print("alpha_beta_search: min_value: to be completed by student")
        return v

    # Body of alpha_beta_search:
    #print("alpha_beta_search: to be completed by students")
    return max(game.actions(state), key=lambda a: min_value(game.result_cutoff(state, a), -np.inf, np.inf, depth))
    


# ______________________________________________________________________________
# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    if (game.d == -1):
        return alpha_beta_search(game, state)
    return alpha_beta_cutoff_search(game, state)


def minmax_player(game,state):
    if( game.d == -1):
        return minmax(game, state)
    return minmax_cutoff(game, state)


def expect_minmax_player(game, state):
    if (game.d == -1):
        return expect_minmax(game, state)
    return expect_minmax_cutoff(game, state)


# ______________________________________________________________________________
# 


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))



class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, h=3, v=3, k=3, d=-1):
        self.h = h
        self.v = v
        self.k = k
        self.depth = d
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)
        
    def result_cutoff(self, state, move): # result function but using evaluation instead of utility
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.evaluation_func(state),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1), self.k) or
                self.k_in_row(board, move, player, (1, 0), self.k) or
                self.k_in_row(board, move, player, (1, -1), self.k) or
                self.k_in_row(board, move, player, (1, 1), self.k)):
            return self.k if player == 'X' else -self.k
        else:
            return 0

    def evaluation_func(self, state):
        """computes value for a player on board after move.
            Likely it is better to conside the board's state from 
            the point of view of both 'X' and 'O' players and then subtract
            the corresponding values before returning."""
        """+100, +10, +1 for 3, 2, 1 in a line for computer
        -100, -10, -1 for 3, 2, 1 in a line for opponent"""
        """Start with 1, each increase in number increases points by x10. All they way to k"""
        
        score = 0;
        # start with X (player), the do O (computer)
        # list(state.board.keys())[0] (most recent move)
        for index in range(self.k):
            
            if self.k_in_row(state.board, list(state.board.keys())[0], 'X', (0, 1), index + 1):
                score += 10 ** index;
            if self.k_in_row(state.board, list(state.board.keys())[0], 'X', (1, 0), index + 1):
                score += 10 ** index;
            if self.k_in_row(state.board, list(state.board.keys())[0], 'X', (1, -1), index + 1):
                score += 10 ** index;
            if self.k_in_row(state.board, list(state.board.keys())[0], 'X', (1, 1), index + 1):
                score += 10 ** index;
            if self.k_in_row(state.board, list(state.board.keys())[0], 'O', (0, 1), index + 1):
                score -= 10 ** index;
            if self.k_in_row(state.board, list(state.board.keys())[0], 'O', (1, 0), index + 1):
                score -= 10 ** index;
            if self.k_in_row(state.board, list(state.board.keys())[0], 'O', (1, -1), index + 1):
                score -= 10 ** index;
            if self.k_in_row(state.board, list(state.board.keys())[0], 'O', (1, 1), index + 1):
                score -= 10 ** index;
                
        # This will now be done in utility() function
        #if self.to_move == 'O': # if it is players turn, score should be inverse of computer
        #    score = -score;
        
        #print("Score is " + str(score))
        #print("evaluation_function: to be completed by students")
        return score;
		
    def k_in_row(self, board, move, player, delta_x_y, number):
        """Return true if there is a line through move on board for player.
        hint: This function can be extended to test of n number of items on a line 
        not just self.k items as it is now. """
        # number is the number in a row it is checking for
        # EX: if number is 2, check for 2 in a row and return true if found
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= number;
        #return n >= self.k
        
    def chances(self, state):
        """Return a list of all possible states."""
        #print("To be completed by students")
        
        """Player is most likely smart
        Split highest odds to best move(s)
            50% they make best move(s)
        Split lowest odds to worst move(s)
            0.1% they make worst move(s)
        Split rest evenly
        
        Evaluation stored in utility so can use utility for any depth
        
        Total = 50% + 0.1% + rest
        rest = 49.9/(n-best&worst)
        """
        
        chances = []
        num_chances = len(self.actions(state))
        
        evals = []
        for a in self.actions(state): # make a list of every evaluation
            evals.append(-self.utility(state, self.to_move(state))) # make them all positive QOL
        
        if not evals:
            return chances
        
        highestEval = max(evals);
        lowestEval = min(evals);
        
        if highestEval == lowestEval: # all evaluations are the same: all chances the same
            for a in self.actions(state):
                chance = 1/num_chances
                chances.append(chance)
            return chances
        
        numOfBest = evals.count(highestEval)
        numOfWorst = evals.count(lowestEval)
        
        for a in self.actions(state):
            
            eval = -self.utility(state, self.to_move(state))
            
            if eval == highestEval:
                chance = 0.5/numOfBest
            elif eval == lowestEval:
                chance = 0.001/numOfWorst
            else:
                chance = 0.499/(num_chances - numOfBest - numOfWorst)
            
            chances.append(chance)
        return chances


class Gomoku(TicTacToe):
    """Also known as Five in a row."""

    def __init__(self, h=15, v=16, k=5):
        TicTacToe.__init__(self, h, v, k)
