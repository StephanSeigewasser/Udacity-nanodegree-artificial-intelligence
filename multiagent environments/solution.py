from copy import deepcopy

X_DIM = 3
Y_DIM = 2

# The eight movement directions possible
DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, -1),
              (-1, 0), (-1, 1), (0, 1), (1, 1)]


class GameState:
    """
    Attributes
    ----------
    TODO: Copy in your implementation from the previous quiz
    """

    def __init__(self):
        # TODO: Copy in your implementation from the previous quiz
        self.board = [[0, 0], [0, 0], [0, 1]]

        self.active_players_index = 0
        self.player_locations = [None, None]

    def actions(self):
        """ Return a list of legal actions for the active player

        You are free to choose any convention to represent actions,
        but one option is to represent actions by the (row, column)
        of the endpoint for the token. For example, if your token is
        in (0, 0), and your opponent is in (1, 0) then the legal
        actions could be encoded as (0, 1) and (0, 2).
        """
        return self.liberties(self.player_locations[self.active_players_index])

    def player(self):
        """ Return the id of the active player

        Hint: return 0 for the first player, and 1 for the second player
        """
        return self.active_players_index

    def result(self, action):
        """ Return a new state that results from applying the given
        action in the current state

        Hint: Check out the deepcopy module--do NOT modify the
        objects internal state in place
        """
        assert action in self.actions(), "Attempted forecast of illegal move"

        nextGameState = deepcopy(self)
        nextGameState.board[action[0]][action[1]] = 1
        nextGameState.player_locations[self.active_players_index] = action
        nextGameState.active_players_index = 1 if self.active_players_index == 0 else 0

        return nextGameState

    def terminal_test(self):
        """ return True if the current state is terminal,
        and False otherwise

        Hint: an Isolation state is terminal if _either_
        player has no remaining liberties (even if the
        player is not active in the current state)
        """
        return (not self.liberties(self.player_locations[0]) or
                not self.liberties(self.player_locations[1]))

    def liberties(self, loc):
        """ Return a list of all open cells in the
        neighborhood of the specified location.  The list
        should include all open spaces in a straight line
        along any row, column or diagonal from the current
        position. (Tokens CANNOT move through obstacles
        or blocked squares in queens Isolation.)

        Note: if loc is None, then return all empty cells
        on the board
        """
        moves = []

        if loc is None:  # return all empty fields
            return [(x, y) for x in range(X_DIM) for y in range(Y_DIM) if self.board[x][y] == 0]

        for dx, dy in DIRECTIONS:  # check each movement direction
            current_x, current_y = loc

            while 0 <= current_x + dx < X_DIM and 0 <= current_y + dy < Y_DIM:
                current_x, current_y = current_x + dx, current_y + dy

                if self.board[current_x][current_y]:  # stop at any blocked cell
                    break

                moves.append((current_x, current_y))

        return moves
