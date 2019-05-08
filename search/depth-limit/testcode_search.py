
import search
import gamestate as game


# Test the depth limit by checking the number of nodes visited
# -- recall that minimax visits every node in the search tree,
# so if we search depth one on an empty board then minimax should
# visit the sum of each sub-tree
depth_limit = 2
expected_node_count = 30
rootNode = game.GameState()
search.get_action(rootNode, depth_limit)

print("Expected node count: {}".format(expected_node_count))
print("Your node count: {}".format(game.call_counter))

if game.call_counter == expected_node_count:
    print("That's right! Looks like your depth limit is working!")
else:
    print("Uh oh...looks like there may be a problem.")
