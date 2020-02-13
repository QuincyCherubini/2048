from board import board
import numpy as np
import math
import time
import random
import cProfile
import keras
from keras.models import load_model
from DQN import DQN
from DQN import create_state

class Node:

    def __init__(self, state, player_turn, DQN, move=9, parent=None):

        self.state = state  # this is a board
        self.parent = parent  # this is a node
        self.children = []  # this is a list of nodes
        self.visits = 0
        self.tot_reward = 0
        self.player_turn = player_turn
        self.move = move
        self.DQN = DQN

    def get_UCB(self):
        # player 1 has a negative score so board score needs to be added instead of subtracted
        if self.player_turn == 1:
            child_avg1 = int(self.tot_reward / self.visits + self.parent.state.score)
        else:
            child_avg1 = int(self.tot_reward / self.visits - self.parent.state.score)

        par_avg1 = int(abs(self.parent.tot_reward / self.parent.visits) - self.parent.state.score)
        exp_ind1 = math.sqrt(math.log2(self.parent.visits) / self.visits)

        UCB = int(child_avg1 + par_avg1*exp_ind1)
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

        # add a random tile if the player is 2 and then play rest of game
        if self.player_turn == 2:
            temp_board.add_new_tile()

            while not temp_board.game_is_over():

                # call the DQN and perform the best move
                cur_state = create_state(temp_board)
                cur_state = cur_state.reshape(1, self.DQN.state_shape[1])
                move = self.DQN.act(cur_state)

                temp_board.move_tiles(move)
                temp_board.add_new_tile()

            # back prop the result of the game
            self.back_prop(temp_board.score)

        # if the player is 1
        elif self.player_turn == 1:

            cur_state = create_state(temp_board)
            cur_state = cur_state.reshape(1, self.DQN.state_shape[1])
            predictions = self.DQN.model.predict(cur_state)[0]

            score = max(predictions)
            self.back_prop(score)


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

                    child = Node(temp_board, 2, self.DQN, move, self)
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

                        child_2 = Node(temp_board_2, 1, self.DQN, 9, self)
                        child_4 = Node(temp_board_4, 1, self.DQN, 9, self)

                        self.children.append(child_2)
                        self.children.append(child_4)

    # note this should only be run on player = 1
    def get_best_move(self):

        max_visits = 0
        max_move = 9
        max_score = 0

        for child in self.children:
            if child.visits > max_visits:
                max_visits = child.visits
                max_move = child.move
                max_score = child.tot_reward/child.visits
            elif child.visits == max_visits:
                if child.tot_reward/child.visits > max_score:
                    max_visits = child.visits
                    max_move = child.move
                    max_score = child.tot_reward / child.visits

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
    # todo: maybe instead of doing a full roll out on each "oppenent" move use the result of the DQN instead?
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

    while time.time() - time_start <= max_time and test_node.visits < max_sims:

        to_exp_child_ind = test_node.get_max_child_UCB()

        if to_exp_child_ind != 99999:
            to_exp_child = test_node.children[to_exp_child_ind]
            take_next_step(to_exp_child)
        else:
            break


def run(max_time, max_turns, max_sims, model):

    # Create a new board and display it
    test_board = board()
    test_board.display()

    # Load the NN from DQN
    DQN_agent = DQN(test_board)
    DQN_agent.model = model

    turns = 0
    while not test_board.game_is_over() and turns < max_turns:

        # create a new node based on the board
        test_node = Node(test_board, 1, DQN_agent, 9, None)

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

        # Test what my Q prediction is
        cur_state = create_state(test_board)
        cur_state = cur_state.reshape(1, DQN_agent.state_shape[1])
        prediction = DQN_agent.model.predict(cur_state)[0]
        print("prediction: {}".format(prediction))

        print(" turn: {} action: {} move: {} sims: {}".format(turns, next_move, move, test_node.visits))

        for child in test_node.children:
            print("child {} visits: {} avg: {} UCB: {}".format(child.move, child.visits,
                        int(child.tot_reward/child.visits - test_board.score), int(child.get_UCB())))

        # Make the move
        test_board.move_tiles(next_move)
        test_board.add_new_tile()
        test_board.display()
        turns += 1


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    max_time = 20  # in seconds
    max_turns = 9999999  # this is used for Testing purposes only
    max_sims = 1000  # maximum number of DQN simulations performed
    model = keras.models.load_model("./trial-3002--3486.model")  # load the DQN model from the save file
    run( max_time, max_turns, max_sims, model)
    # pr.disable()
    # pr.print_stats()
