import random, math, time
from sample_players import DataPlayer
from isolation import DebugState

# board
_WIDTH = 11
_HEIGHT = 9

# minimax with alpha-beta-pruning
_SEARCH_DEPTH = 5
_CENTER = [5, 4]

# monte carlo tree search
_EXPLORATION_WEIGHT = 1.7
_SEARCH_TIME = 150
_SEARCH_TIME_WITH_SAFETY = 0.5


# Representation of a node for the monte carlo tree search.
# Provides several helper methods to traverse through the tree.
class Node:
    def __init__(self, gameState, action=None, parent=None):
        self.action = action
        self.parentNode = parent
        self.gameState = gameState
        self.childNodes = []
        self.qValue = 0
        self.numberOfVisits = 0
        self.openActions = gameState.actions()

    def is_fully_expanded(self):
        return len(self.openActions) == 0

    def expand(self):
        move = random.choice(self.openActions)
        self.openActions.remove(move)
        next_state = self.gameState.result(move)
        child_node = Node(next_state, move, self)
        self.childNodes.append(child_node)

        return child_node

    def best_child(self):
        max_uct = float("-inf")
        best_child = None

        for childNode in self.childNodes:
            uct = (childNode.qValue / childNode.numberOfVisits) + _EXPLORATION_WEIGHT * math.sqrt(
                2 * math.log(self.numberOfVisits) / childNode.numberOfVisits)

            if uct > max_uct:
                max_uct = uct
                best_child = childNode

        return best_child

    # default policy
    def rollout(self):
        current_game_state = self.gameState

        while Node.non_terminal(current_game_state):
            next_action = Node.rollout_policy(current_game_state.actions())
            current_game_state = current_game_state.result(next_action)

        return self.result(current_game_state)

    def backpropagate(self, result):
        self.numberOfVisits += 1.
        self.qValue += result

        result = -result
        if self.parentNode:
            self.parentNode.backpropagate(result)

    @staticmethod
    def non_terminal(game_state):
        return not game_state.terminal_test()

    @staticmethod
    def rollout_policy(actions):
        return random.choice(actions)

    @staticmethod
    def result(game_state):
        if Node.is_a_win(game_state):
            return 1.
        elif Node.is_a_loss(game_state):
            return -1.
        else:
            return 0

    @staticmethod
    def is_a_win(game_state):
        return game_state.utility(1 - game_state.player()) == float('inf')

    @staticmethod
    def is_a_loss(game_state):
        return game_state.utility(1 - game_state.player()) == float('-inf')


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
            best_action = self.alpha_beta_search(state, _SEARCH_DEPTH)
            # best_action = CustomPlayer.monte_carlo_tree_search(state)

            self.queue.put(best_action)

    @staticmethod
    def monte_carlo_tree_search(game_state=None, search_time=_SEARCH_TIME):
        root_node = Node(game_state)

        start_time = time.time()

        # remain a little safety
        search_time = math.ceil(search_time * _SEARCH_TIME_WITH_SAFETY)

        while (time.time() - start_time) * 1000 <= search_time:
            leaf_node = CustomPlayer.traverse(root_node)
            simulation_result = leaf_node.rollout()
            leaf_node.backpropagate(simulation_result)

        best_child = root_node.best_child()
        return best_child.action

    # tree policy
    @staticmethod
    def traverse(node):
        node_to_process = node

        while Node.non_terminal(node_to_process.gameState):
            if not node_to_process.is_fully_expanded():
                return node_to_process.expand()
            else:
                node_to_process = node_to_process.best_child()

        return node_to_process

    def alpha_beta_search(self, game_state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in game_state.actions():
            v = self.min_value(game_state.result(a), alpha, beta, depth - 1)
            alpha = max(alpha, v)

            if v > best_score:
                best_score = v
                best_move = a
            elif best_move is None:
                # take at least an action to don't get stuck
                best_score = v
                best_move = a

        return best_move

    def min_value(self, game_state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if game_state.terminal_test():
            return game_state.utility(self.player_id)

        if depth <= 0:
            return self.score(game_state)

        v = float("inf")

        for a in game_state.actions():
            v = min(v, self.max_value(game_state.result(a), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def max_value(self, game_state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if game_state.terminal_test():
            return game_state.utility(self.player_id)

        if depth <= 0:
            return self.score(game_state)

        v = float("-inf")

        for a in game_state.actions():
            v = max(v, self.min_value(game_state.result(a), alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]

        # diff_liberties
        return self.diff_liberties(state)

        # own center
        # return 5 - CustomPlayer.center_field_heuristic(own_loc, state)

        # opp border
        # return CustomPlayer.center_field_heuristic(opp_loc, state) - 5

        # own center + 2 * diff_liberties
        # return 5 - CustomPlayer.center_field_heuristic(own_loc, state) + 2 * self.diff_liberties(state)

        # own center + diff_liberties
        # return 5 - CustomPlayer.center_field_heuristic(own_loc, state) + self.diff_liberties(state)

        # opp border - own center
        # return CustomPlayer.center_field_heuristic(opp_loc, state) - CustomPlayer.center_field_heuristic(own_loc, state)

        # opp border - own center + diff_liberties
        # return CustomPlayer.center_field_heuristic(opp_loc, state) - CustomPlayer.center_field_heuristic(own_loc, state) + self.diff_liberties(state)

        # own free fields
        # return CustomPlayer.empty_surrounding_fields_heuristic(own_loc, state)

        # own free fields - opp free fields
        # return CustomPlayer.empty_surrounding_fields_heuristic(own_loc, state) - CustomPlayer.empty_surrounding_fields_heuristic(opp_loc, state)

        # own free fields + own center
        # return self.empty_surrounding_fields_heuristic(own_loc, state) + 6 - self.center_field_heuristic(own_loc)

        # minimize distance to opp
        # return self.minimize_distance_to_opponent_heuristic(state)

        # maximize distance to opp
        # return self.maximize_distance_to_opponent_heuristic(state)

        # biggest quadrant
        # return CustomPlayer.biggest_quadrant_heuristic(own_loc, state)

        # own biggest - opp biggest
        # return CustomPlayer.biggest_quadrant_heuristic(own_loc, state) - CustomPlayer.biggest_quadrant_heuristic(opp_loc, state)

    def diff_liberties(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        return len(own_liberties) - len(opp_liberties)

    def minimize_distance_to_opponent_heuristic(self, game_state):
        own_loc = game_state.locs[self.player_id]
        opp_loc = game_state.locs[1 - self.player_id]

        debug_state = DebugState.from_state(game_state)
        own_xy_position = debug_state.ind2xy(own_loc)
        opp_xy_position = debug_state.ind2xy(opp_loc)

        distance = CustomPlayer.xy_distance(own_xy_position, opp_xy_position)

        # max distance is 9, so calculate a score which is positive for lower distance and negative for bigger one
        return 9 - distance

    def maximize_distance_to_opponent_heuristic(self, game_state):
        own_loc = game_state.locs[self.player_id]
        opp_loc = game_state.locs[1 - self.player_id]

        debug_state = DebugState.from_state(game_state)
        own_xy_position = debug_state.ind2xy(own_loc)
        opp_xy_position = debug_state.ind2xy(opp_loc)

        distance = CustomPlayer.xy_distance(own_xy_position, opp_xy_position)

        # max distance is 18, so calculate a score which is positive for bigger distance and negative for lower one
        return distance - 9

    @staticmethod
    def center_field_heuristic(position, game_state):
        xy_position = DebugState.from_state(game_state).ind2xy(position)

        return CustomPlayer.xy_distance(xy_position, _CENTER)

    @staticmethod
    def empty_surrounding_fields_heuristic(position, game_state):
        empty_fields = 0

        for x_modifier in range(-2, 3):
            for y_modifier in range(-2, 3):
                # board width is 11 + 2 = 13 because of the boarders left and right, therefore the modifier is n*13
                modifier = x_modifier + (_WIDTH + 2) * y_modifier
                field_pos = position + modifier

                if field_pos >= 0:
                    empty_fields += CustomPlayer.check_if_field_is_empty(field_pos, game_state)

        print("got ", empty_fields, " for position ", position)
        return empty_fields

    @staticmethod
    def biggest_quadrant_heuristic(position, game_state):
        debug_state = DebugState.from_state(game_state)

        xy_position = debug_state.ind2xy(position)

        q1_xy_modifiers = CustomPlayer.count_modifiers_for_q1(xy_position)
        q1_empty_fields = CustomPlayer.count_empty_fields(position, game_state, q1_xy_modifiers)

        q2_xy_modifiers = CustomPlayer.count_modifiers_for_q2(xy_position)
        q2_empty_fields = CustomPlayer.count_empty_fields(position, game_state, q2_xy_modifiers)

        q3_xy_modifiers = CustomPlayer.count_modifiers_for_q3(xy_position)
        q3_empty_fields = CustomPlayer.count_empty_fields(position, game_state, q3_xy_modifiers)

        q4_xy_modifiers = CustomPlayer.count_modifiers_for_q4(xy_position)
        q4_empty_fields = CustomPlayer.count_empty_fields(position, game_state, q4_xy_modifiers)

        # max empty fields is nearly 100, so calculate a score which is positive for bigger distance and negative for lower one
        return max(q1_empty_fields, q2_empty_fields, q3_empty_fields, q4_empty_fields) - 40

    @staticmethod
    def count_empty_fields(position, game_state, xy_modifiers):
        empty_fields = 0

        for x_modifier in range(0, xy_modifiers[0]):
            for y_modifier in range(0, xy_modifiers[1]):
                modifier = x_modifier + (_WIDTH + 2) * y_modifier
                field_pos = position + modifier

                empty_fields += CustomPlayer.check_if_field_is_empty(field_pos, game_state)

        return empty_fields

    @staticmethod
    def check_if_field_is_empty(position, game_state):
        try:
            # if bit shift leads to the same number the field is empty
            if game_state.board & (1 << position):
                return 1
        except IndexError:
            # do nothing
            pass

        return 0

    @staticmethod
    def count_modifiers_for_q1(xy_position):
        # compute Q1: 11-x <-> x, 9-y <-> y
        return [_WIDTH - xy_position[0], _HEIGHT - xy_position[1]]

    @staticmethod
    def count_modifiers_for_q2(xy_position):
        # compute Q2: x <-> x+1, 9-y <-> y
        return [xy_position[0] + 1, _HEIGHT - xy_position[1]]

    @staticmethod
    def count_modifiers_for_q3(xy_position):
        # compute Q3: x <-> x+1, y <-> y+1
        return [xy_position[0] + 1, xy_position[1] + 1]

    @staticmethod
    def count_modifiers_for_q4(xy_position):
        # compute Q4: 11-x <-> x, y <-> y+1
        return [_WIDTH - xy_position[0] + 1, xy_position[1] + 1]

    @staticmethod
    def xy_distance(position_1, position_2):
        return abs(position_1[0] - position_2[0]) + abs(position_1[1] - position_2[1])
