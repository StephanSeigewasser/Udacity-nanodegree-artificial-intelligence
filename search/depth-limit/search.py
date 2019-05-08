
from minimax import minimax_decision

def get_action(gameState, depth_limit):
    # TODO: Implement a function that calls minimax_decision
    # for each depth from 1...depth_limit (inclusive of both endpoints)
    best_move = None
    for depth in range(1, depth_limit+1):
        best_move = minimax_decision(gameState, depth)
    return best_move