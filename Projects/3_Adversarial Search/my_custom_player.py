import random
import time
from sample_players import DataPlayer
#from isolation.isolation import _WIDTH, _HEIGHT, _SIZE
import isolation as isolation
from collections import defaultdict
import sys, math

#CENTER = (_WIDTH*_HEIGHT-3)/2
#xPenalty = {0:0, 1:0,2:0, 3:0, 4:0.25,5:0.5, 6:1.0} # Index 6 is not required for game play, it is required for my testing.
#yPenalty = {0:0, 1:0,2:0, 3:0.25,4:0.5}
#node_counter = 0

# https://int8.io/monte-carlo-tree-search-beginners-guide/
# http://mcts.ai/about/

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
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            ''' Monte Carlo Tree Search '''
            mcts = MonteCarloTreeSearch(IsolationTreeNode(state), bias_param_const = 0.5)
            while True:
                mcts.run()
                best_action = mcts.best_action()
                #print("Based on ", node_counter, " nodes")
                self.queue.put(best_action)

            ''' Iterative Alpha Beta Prunning Search '''
            '''for depth in range(3, 10):  # Minimax plays with depth 3, so this depth is always possible
                best_action = self.alpha_beta_search(state, depth)
                self.queue.put(best_action)'''


    def alpha_beta_search(self, state, depth):
        def min_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta = float("inf")
        best_value = float("-inf")  # This is actually a duplicate of alpha : can be removed
        best_action = None
        for action in state.actions():
            value = min_value(state.result(action), alpha, beta, depth-1)
            alpha = max(alpha, value)
            if value >= best_value:
                best_value = value
                best_action = action
        return best_action

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        #intersect = set(own_liberties).intersection(opp_liberties)
        #return len(own_liberties) - len(opp_liberties) + len(intersect)
        #return len(own_liberties) - len(intersect)
        #dy = round((own_loc-57)/13)
        #dx = round(own_loc - 13*dy - 57)
        #distance_penalty = xPenalty[abs(dx)] + yPenalty[abs(dy)]
        # print("dist {}, dx {}, dy {}, own_loc {}".format(distance_from_centre, dx, dy, own_loc))

        return len(own_liberties) - len(opp_liberties) # - distance_penalty

class IsolationTreeNode():
    def __init__(self, state: isolation, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.untried_actions = self.state.actions()
        self.total_reward = 0
        self.num_visits = 0
        #global node_counter
        #node_counter += 1

    ''' Expand using random untried action '''
    def expand(self):
        #print("IsolationTreeNode.expand")
        #print(len(self.untried_actions), " Should be more than 0")
        next_action = self.untried_actions.pop(random.randint(0, len(self.untried_actions)-1))
        new_child = IsolationTreeNode(self.state.result(next_action), parent=self)
        # print(next_action, self.state, new_child.state)
        self.children[new_child] = next_action
        return new_child

    ''' Best child with highest UCT best_score
        UCT - choice based on exploration and exploitation balance '''
    def best_uct(self, bias_param):
        #print("IsolationTreeNode.best_uct")
        maxUCT = float('-inf')
        bestChild = None
        for child in self.children:
            uct = child.total_reward/child.num_visits + bias_param * math.sqrt(math.log(self.num_visits)/child.num_visits)
            if uct > maxUCT:
                bestChild = child
                maxUCT = uct
        return bestChild

    ''' Best action based on the results so far '''
    def best_action(self):
        # return max(child.num_visits for child in self.children)
        best_child = max(self.children.keys(), key = lambda item: item.num_visits)
        return self.children[best_child]

    ''' is Leaf node '''
    def is_terminal(self):
        #print("IsolationTreeNode.is_terminal")
        return self.state.terminal_test()

class MonteCarloTreeSearch():
    def __init__(self, state:IsolationTreeNode, bias_param_const):
        # print("MonteCarloTreeSearch.init")
        self.root = state
        self.bias_param_const = bias_param_const

    ''' Choose next node for simulation '''
    def tree_policy(self):
        # print("MonteCarloTreeSearch.tree_policy")
        current_node = self.root
        while not current_node.is_terminal():
            if current_node.untried_actions:
                return current_node.expand()
            else:
                current_node = current_node.best_uct(bias_param = self.bias_param_const)
        return current_node

    ''' Rollout policy to choose from available actions '''
    def rollout_policy(self, available_actions):
        # print("MonteCarloTreeSearch.rollout_policy")
        return random.choice(available_actions)

    ''' Parent of node chooses reward value when choosing next action
        Win for this node is lose for parent and vice-versa '''
    def reward_value_for_parent(self, state):
        # print("MonteCarloTreeSearch.reward_value_for_parent")
        if state.utility(state.player())==float('inf'):
            return -1.0 # Negative reward for Win
        elif state.utility(state.player())==float('-inf'):
            return 1.0  # Positive reward for Loss
        return 0.0      # Draw

    ''' Complete one game/ simulation till terminal state '''
    def simulate(self, start_node):
        # print("MonteCarloTreeSearch.simulate")
        state = start_node.state
        # print(state)
        while not state.terminal_test():
            state = state.result(self.rollout_policy(state.actions()))
        return self.reward_value_for_parent(state)

    ''' Backpropogate the reward to parents till root '''
    def backpropagate(self, node, reward):
        #print("MonteCarloTreeSearch.backpropagate")
        node.num_visits += 1
        node.total_reward += reward
        if node.parent:
            self.backpropagate(node.parent, -reward)

    ''' Run Monte Carlo Tree Search Simulation '''
    def run(self):
        #print("MonteCarloTreeSearch.run")
        start_node = self.tree_policy()
        reward = self.simulate(start_node)
        # print("Start State = ", start_node.state, " Reward = ", reward )
        self.backpropagate(start_node, reward)

    ''' Return the best action from the simulations run so far '''
    def best_action(self):
        # print("MonteCarloTreeSearch.best_action")
        # best_child_node = self.root.best_uct(bias_param = 0) # Exploitation Only
        return self.root.best_action()
