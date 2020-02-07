from board import board
import numpy as np
import math
import time
import random
import cProfile

class Node:

    def __init__(self, state, player_turn, move=9, parent=None, expansion=2):

        self.state = state  # this is a board
        self.parent = parent  # this is a node
        self.children = []  # this is a list of nodes
        self.visits = 0
        self.tot_reward = 0
        self.expansion = expansion
        self.player_turn = player_turn
        self.move = move

    def get_UCB(self):
        UCB = self.tot_reward/self.visits - self.state.score + self.expansion*math.sqrt(math.log2(self.parent.visits) / self.visits)
        return UCB

    def is_leaf(self):
        return len(self.children) == 0

    def is_unchecked(self):
        return self.visits == 0

    def get_min_diff(self):

        max_visits = 0
        max_child = None
        for child in self.children:
            if child.visits > max_visits:
                max_visits = child.visits
                max_child = child

        min_diff = 99999999999
        for child in self.children:
            if max_visits - child.visits < min_diff and child != max_child:
                min_diff = max_visits - child.visits

        return min_diff

    def get_max_child_UCB(self):

        max_UCB = -999999999
        max_child_index = 99999

        for child in self.children:
            if child.get_UCB() > max_UCB:  # and not child.is_terminal_node():
                max_UCB = child.get_UCB()
                max_child_index = self.children.index(child)

        if max_child_index != 99999:
            return max_child_index
        else:
            print("INDEX OUT OF RANGE: max_UCB: {} max_child_index: {}".format(max_UCB, max_child_index))
            print("len(self.children): {}".format(len(self.children)))
            print("player_turn: {}".format(self.player_turn))
            print("is_leaf(): {}".format(self.is_leaf()))
            print("is_terminal_node(): {}".format(self.is_terminal_node()))

            for child in self.children:
                print("get_UCB(): {} is_terminal_node: {}".format(child.get_UCB(), child.is_terminal_node()))

            return max_child_index

    def back_prop(self, score):
        self.visits += 1

        if self.player_turn == 1:
            self.tot_reward -= score
        else:
            self.tot_reward += score

        if self.parent is not None:
            self.parent.back_prop(score)

    def rollout(self):

        # test_board = self.state
        temp_board = board()
        temp_board.score = self.state.score
        for x in [0, 1, 2, 3]:
            for y in [0, 1, 2, 3]:
                temp_board.tiles[x][y] = self.state.tiles[x][y]

        while True:

            # if the game is over exit the loop and back prop
            if temp_board.game_is_over():  # or turn_count == 30:
                self.back_prop(temp_board.score)
                break

            # otherwise perform a random legal move
            move = random.getrandbits(2)
            while not temp_board.is_valid_move(move):
                move = (move + 1) % 4

            temp_board.move_tiles(move)
            temp_board.add_new_tile()

    def expand(self):

        # go through 4 moves here
        if self.player_turn == 1:
            for move in [0, 1, 2, 3]:

                # if the move is legal add a child node to the child array
                if self.state.is_valid_move(move):

                    test_board = self.state
                    temp_board = board()
                    temp_board.score = test_board.score
                    for x in [0, 1, 2, 3]:
                        for y in [0, 1, 2, 3]:
                            temp_board.tiles[x][y] = test_board.tiles[x][y]
                    temp_board.move_tiles(move)

                    child = Node(temp_board, 2, move, self, self.expansion)
                    self.children.append(child)

        # add random tile to each spot
        else:
            for x in [0, 1, 2, 3]:
                for y in [0, 1, 2, 3]:
                    if self.state.tiles[x][y] == 0:

                        test_board = self.state

                        temp_board_2 = board()
                        temp_board_2.score = test_board.score

                        temp_board_4 = board()
                        temp_board_4.score = test_board.score

                        for r in [0, 1, 2, 3]:
                            for q in [0, 1, 2, 3]:
                                temp_board_2.tiles[r][q] = test_board.tiles[r][q]
                                temp_board_4.tiles[r][q] = test_board.tiles[r][q]

                        temp_board_2.tiles[x][y] = 2
                        temp_board_4.tiles[x][y] = 4

                        child_2 = Node(temp_board_2, 1, 9, self, self.expansion)
                        child_4 = Node(temp_board_4, 1, 9, self, self.expansion)

                        self.children.append(child_2)
                        self.children.append(child_4)

    # note this should only be run on player = 1
    def get_best_move(self):

        max_visits = 0
        max_move = 9

        for child in self.children:
            if child.visits > max_visits:
                max_visits = child.visits
                max_move = child.move

        if max_move != 9:
            return max_move
        else:
            print("THIS SHOULD NEVER HAPPEN!!!!!!!!!!! max_score: {}".format(max_score))

    def is_terminal_node(self):
        return self.state.game_is_over()

def take_next_step(test_node):

    # if the test_node is a leaf expand it
    if test_node.is_leaf():
        test_node.expand()

    # check if any of the children are unexplored, if so explore them
    all_children_checked = True
    for child in test_node.children:
        if child.is_unchecked():
            child.rollout()
            all_children_checked = False

    if all_children_checked:
        if not test_node.is_terminal_node():
            to_exp_child_ind = test_node.get_max_child_UCB()
            to_exp_child = test_node.children[to_exp_child_ind]
            take_next_step(to_exp_child)
        else:
            test_node.rollout()


def expand_node(test_node, time_start, max_time, max_sims):

    # if the test_node is a leaf expand it
    if test_node.is_leaf():
        test_node.expand()

    # check if any of the children are unexplored, if so explore them
    for child in test_node.children:
        if child.is_unchecked():
            child.rollout()

    total_visits = 0

    while time.time() - time_start <= max_time and test_node.visits < max_sims:

        to_exp_child_ind = test_node.get_max_child_UCB()

        if to_exp_child_ind != 99999:
            to_exp_child = test_node.children[to_exp_child_ind]
            take_next_step(to_exp_child)
        else:
            break

    # print(sims)


def run(exploration_num, max_time, max_turns, max_sims):

    # Create a new board and display it
    test_board = board()
    test_board.display()

    turns = 0
    while not test_board.game_is_over() and turns < max_turns:

        # get node
        # create a new node based on the board
        test_node = Node(test_board, 1, 9, None, exploration_num)

        # expand the tree while I have time
        time_start = time.time()
        expand_node(test_node, time_start, max_time, max_sims)

        # pick the next best move
        next_move = test_node.get_best_move()

        if next_move == 0:
            move = "left"
        elif next_move == 1:
            move = "down"
        elif next_move == 2:
            move = "right"
        elif next_move == 3:
            move = "up"

        print(" turn: {} action: {} move: {} sims: {}".format(turns, next_move, move, test_node.visits))

        counter = 0
        for child in test_node.children:
            print("child {} visits: {} total: {} avg: {} UCB: {}".format(child.move, child.visits, child.tot_reward,
                                            child.tot_reward/child.visits - test_board.score, child.get_UCB()))
            counter += 1

        # Make the move
        test_board.move_tiles(next_move)
        test_board.add_new_tile()
        test_board.display()
        turns += 1


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    exploration_num = 750  # todo: is this the best number?
    max_time = .5  # in seconds
    max_turns = 9999999  # this is used for Testing purposes only
    max_sims = 500  #
    run(exploration_num, max_time, max_turns, max_sims)
    pr.disable()
    pr.print_stats()
