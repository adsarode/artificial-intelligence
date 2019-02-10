import random
import time
from sample_players import DataPlayer
from isolation.isolation import _WIDTH, _HEIGHT, _SIZE
TIME_LIMIT = 150
SIMULATION_TIME = TIME_LIMIT*0.80
CENTER = (_WIDTH*_HEIGHT-3)/2

xPenalty = {0:0, 1:0,2:0, 3:0, 4:0.25,5:0.5, 6:1.0} # Index 6 is not required for game play, it is required for my testing.
yPenalty = {0:0, 1:0,2:0, 3:0.25,4:0.5}

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        start_time = time.time()*1000
        stop_time = start_time + SIMULATION_TIME

        # print(SIMULATION_TIME)

        # self.queue.put(random.choice(state.actions()))
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            # self.queue.put(self.minimax(state, depth=3))
            # self.queue.put(self.alpha_beta_search(state, depth=3))
            move, score = self.alpha_beta_search(state, 3, stop_time)
            self.queue.put(move)
            #self.queue.put(self.iterative_alpha_beta(state, stop_time))

    def iterative_alpha_beta(self, state, stop_time):
        best_score = float("-inf")
        best_move = None
        depth_init = 3
        depth_limit = 10
        for depth in range(depth_init, depth_limit):
            move, score = self.alpha_beta_search(state, depth, stop_time)
            if score >= best_score:
                best_move = move
                best_score = score
            if time.time()*1000 >= stop_time:
                break
        return best_move

    def alpha_beta_search(self, gameState, depth, stop_time):
        def min_value(gameState, alpha, beta, depth):
            if gameState.terminal_test():
                return gameState.utility(self.player_id)
            if depth <= 0: return self.score(gameState)

            v = float("inf")
            for a in gameState.actions():
                #if time.time()*1000 >= stop_time: return v     # Turn on for iterative deepening, can be off otherwise
                v = min(v, max_value(gameState.result(a), alpha, beta, depth-1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(gameState, alpha, beta, depth):
            if gameState.terminal_test():
                return gameState.utility(self.player_id)
            if depth <= 0: return self.score(gameState)

            v = float("-inf")
            for a in gameState.actions():
                #if time.time()*1000 >= stop_time: return v     # Turn on for iterative deepening, can be off otherwise
                v = max(v, min_value(gameState.result(a), alpha, beta, depth-1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in gameState.actions():
            v = min_value(gameState.result(a), alpha, beta, depth-1)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move, best_score

    def minimax(self, state, depth):

        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        #intersect = set(own_liberties).intersection(opp_liberties)
        #return len(own_liberties) - len(opp_liberties) + len(intersect)
        #return len(own_liberties) - len(intersect)
        dy = round((own_loc-57)/13)
        dx = round(own_loc - 13*dy - 57)
        distance_penalty = xPenalty[abs(dx)] + yPenalty[abs(dy)]
        # print("dist {}, dx {}, dy {}, own_loc {}".format(distance_from_centre, dx, dy, own_loc))

        return len(own_liberties) - len(opp_liberties) - distance_penalty
