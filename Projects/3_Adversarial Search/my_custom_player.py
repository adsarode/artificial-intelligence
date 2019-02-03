
from sample_players import DataPlayer


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
        import random
        # self.queue.put(random.choice(state.actions()))
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.minimax(state, depth=3))
            # self.queue.put(self.alpha_beta_search(state, depth=3))

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
        return len(own_liberties) - len(opp_liberties)

    # def score(self, state):
    #     own_loc = state.locs[self.player_id]
    #     opp_loc = state.locs[1 - self.player_id]
    #     own_liberties = state.liberties(own_loc)
    #     opp_liberties = state.liberties(opp_loc)
    #     print("opp_liberties = {} and own_liberties = {}".format(opp_liberties, own_liberties))
    #     # print("opp_liberties = %s and own_liberties = %s" (opp_liberties, own_liberties))
    #     if opp_liberties is 0:
    #         return (own_liberties*2)
    #         print("Killer Move")
    #     elif own_liberties is 0:
    #         return (opp_liberties*-2)
    #         print("Killer Move")
    #     else:
    #        return 1#len(own_liberties) - len(opp_liberties)

    # def score(self, state):
    #     own_loc = state.locs[self.player_id]
    #     opp_loc = state.locs[1 - self.player_id]
    #     own_liberties = len(state.liberties(own_loc))
    #     opp_liberties = len(state.liberties(opp_loc))
    #     if opp_liberties < 2:
    #         return 10#(own_liberties*2)
    #     elif own_liberties < 2:
    #         return (opp_liberties*-2)
    #     else:
    #        return -10#own_liberties - opp_liberties

    def alpha_beta_search(self, gameState, depth):
        def min_value(gameState, alpha, beta, depth):
            if gameState.terminal_test():
                return gameState.utility(self.player_id)
            if depth <= 0: return self.score(gameState)

            v = float("inf")
            for a in gameState.actions():
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
            v = min_value(gameState.result(a), alpha, beta, depth)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
