from board import board
import numpy as np
import math
import time
import random
import pickle
import cProfile
from collections import deque
import keras
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras import initializers
from keras.layers.advanced_activations import LeakyReLU
from MCTS import Node


class DQN:
    # my board is the environment
    def __init__(self, board, exploration_num, max_time):
        self.memory = deque(maxlen=50000)
        self.board = board
        self.state_shape = [1, 160]
        self.gamma = 0.995
        self.epsilon = 0.25
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.005
        self.tau = 0.125
        self.exploration_num = exploration_num
        self.max_time = max_time

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(85, input_dim=self.state_shape[1]))
        model.add(Dense(4))  # Action space for 2048
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 20
        if len(self.memory) < 50:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # return a simulated move
        if np.random.random() < self.epsilon:
            # create a new node based on the board
            test_node = Node(self.board, 1, 9, None, self.exploration_num)

            # expand the tree while I have time
            time_start = time.time()
            expand_node(test_node, time_start, self.max_time)

            action = test_node.get_best_move()

        else:
            output_array = self.model.predict(state)[0]
            action = np.argmax(output_array)
            # if the action is invalid choose the next best
            while not self.board.is_valid_move(action):
                output_array[action] = -999999999999999
                action = np.argmax(output_array)

        return action

    def save_model(self, fn, target_n):
        self.model.save(fn)
        self.target_model.save(target_n)


# Monte Carlo Tree Search functions
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


def expand_node(test_node, time_start, max_time):

    # if the test_node is a leaf expand it
    if test_node.is_leaf():
        test_node.expand()

    # check if any of the children are unexplored, if so explore them
    for child in test_node.children:
        if child.is_unchecked():
            child.rollout()

    while time.time() - time_start <= max_time:

        to_exp_child_ind = test_node.get_max_child_UCB()

        if to_exp_child_ind != 99999:
            to_exp_child = test_node.children[to_exp_child_ind]
            take_next_step(to_exp_child)
        else:
            break


# run MCTS simulations with a 1 second time on each and log the states
def run_initial_MCTS(my_board, dqn_agent, exploration_num, max_time):

    for _ in range(10):
        # Create a new board and display it
        my_board.reset()
        my_board.display()

        # create the current state object
        cur_state = create_state(my_board)
        cur_state = cur_state.reshape(1, dqn_agent.state_shape[1])

        test_state = create_state(my_board)
        test_state = cur_state.reshape(1, dqn_agent.state_shape[1])

        turns = 0
        while not my_board.game_is_over():

            if turns % 100 == 99:
                print("test prediction: {} score: {}".format(dqn_agent.model.predict(test_state)[0], my_board.score))

            # create a new node based on the board
            test_node = Node(my_board, 1, 9, None, exploration_num)

            # expand the tree while I have time
            time_start = time.time()
            expand_node(test_node, time_start, max_time)

            # pick the next best move
            next_move = test_node.get_best_move()

            # Make the move
            cur_score = my_board.score
            my_board.move_tiles(next_move)
            my_board.add_new_tile()
            # my_board.display()

            # log the move data
            reward = my_board.score - cur_score
            new_state = create_state(my_board)
            new_state = new_state.reshape(1, dqn_agent.state_shape[1])
            done = my_board.game_is_over()
            dqn_agent.remember(cur_state, next_move, reward, new_state, done)

            # update the model
            dqn_agent.replay()
            dqn_agent.target_train()

            # update the current state
            cur_state = new_state

            # tracker for output
            turns += 1


def run(episodes, exploration_num, max_time):

    my_board = board()
    dqn_agent = DQN(my_board, exploration_num, max_time)
    test_state = create_state(my_board)
    test_state = test_state.reshape(1, dqn_agent.state_shape[1])

    # run the initial 2 MCTS simulations
    run_initial_MCTS(my_board, dqn_agent, exploration_num, max_time)

    max_score = 0
    total_score = 0

    for episode in range(episodes):

        my_board.reset()
        cur_state = create_state(my_board)
        cur_state = cur_state.reshape(1, dqn_agent.state_shape[1])
        print("test prediction: {}".format(dqn_agent.model.predict(test_state)[0]))

        while not my_board.game_is_over():

            # get the next action
            action = dqn_agent.act(cur_state)

            # perform the action
            cur_score = my_board.score
            my_board.move_tiles(action)
            my_board.add_new_tile()

            reward = my_board.score - cur_score
            new_state = create_state(my_board)
            new_state = new_state.reshape(1, dqn_agent.state_shape[1])
            done = my_board.game_is_over()

            # log this action
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            cur_state = new_state

            # update the model
            dqn_agent.replay()
            dqn_agent.target_train()

        total_score += my_board.score
        if my_board.score > max_score:
            max_score = my_board.score

        print("episode: {} score: {} max_score: {} avg_score: {} epsilon: {}".format(episode, my_board.score, max_score,
                    int(total_score/(episode + 1)), dqn_agent.epsilon))

        if episode % 100 == 2:
            # print("output layer bias: {}".format(dqn_agent.model.layers[3].get_weights()[1]))
            # show_game(dqn_agent)
            dqn_agent.save_model("obj/10/trial-{}--{}.model".format(episode, str(int(total_score/(episode + 1)))),
                                 "obj/10/trial-{}-target.model".format(episode))


def create_state(board):

    output = []

    # give the log base 2 of the board
    for x in range(0, 4):
        for y in range(0, 4):
            # track the value of the tile
            if board.tiles[x][y] == 0:
                output.append(0)
            else:
                output.append(math.log(board.tiles[x][y], 2))

            # track if the tile is empty
            if board.tiles[x][y] == 0:
                output.append(1)
            else:
                output.append(0)

            # Can merge left test
            # if the tile is on the left edge or is empty can't merge
            if x == 0 or board.tiles[x][y] == 0:
                output.append(0)
            else:
                temp_x = x - 1
                while temp_x >= 0:
                    # if the left tile matches the value then append 1
                    if board.tiles[x][y] == board.tiles[temp_x][y]:
                        output.append(1)
                        break
                    # if we get to the edge without finding a match append 0
                    elif temp_x == 0:
                        output.append(0)
                        break
                    temp_x -= 1

            # Can merge right test
            # if the tile is on the left edge or is empty can't merge
            if x == 3 or board.tiles[x][y] == 0:
                output.append(0)
            else:
                temp_x = x + 1
                while temp_x <= 3:
                    # if the left tile matches the value then append 1
                    if board.tiles[x][y] == board.tiles[temp_x][y]:
                        output.append(1)
                        break
                    # if we get to the edge without finding a match append 0
                    elif temp_x == 3:
                        output.append(0)
                        break
                    temp_x += 1

            # Can merge up test
            # if the tile is on the left edge or is empty can't merge
            if y == 0 or board.tiles[x][y] == 0:
                output.append(0)
            else:
                temp_y = y - 1
                while temp_y >= 0:
                    # if the left tile matches the value then append 1
                    if board.tiles[x][y] == board.tiles[x][temp_y]:
                        output.append(1)
                        break
                    # if we get to the edge without finding a match append 0
                    elif temp_y == 0:
                        output.append(0)
                        break
                    temp_y -= 1

            # Can merge down test
            # if the tile is on the left edge or is empty can't merge
            if y == 3 or board.tiles[x][y] == 0:
                output.append(0)
            else:
                temp_y = y + 1
                while temp_y <= 3:
                    # if the left tile matches the value then append 1
                    if board.tiles[x][y] == board.tiles[x][temp_y]:
                        output.append(1)
                        break
                    # if we get to the edge without finding a match append 0
                    elif temp_y == 3:
                        output.append(0)
                        break
                    temp_y += 1

            # if the tile to the left is 2x (directly left only)
            if x == 0 or board.tiles[x][y] == 0:
                output.append(0)
            elif board.tiles[x - 1][y] == 2 * board.tiles[x][y]:
                output.append(1)
            else:
                output.append(0)

            # if the tile to the right is 2x (directly only)
            if x == 3 or board.tiles[x][y] == 0:
                output.append(0)
            elif board.tiles[x + 1][y] == 2 * board.tiles[x][y]:
                output.append(1)
            else:
                output.append(0)

            # if the tile above is 2x (directly only)
            if y == 0 or board.tiles[x][y] == 0:
                output.append(0)
            elif board.tiles[x][y - 1] == 2 * board.tiles[x][y]:
                output.append(1)
            else:
                output.append(0)

            # if the tile below is 2x (directly only)
            if y == 3 or board.tiles[x][y] == 0:
                output.append(0)
            elif board.tiles[x][y + 1] == 2 * board.tiles[x][y]:
                output.append(1)
            else:
                output.append(0)

    # todo: add is next to 2x square (left/up/right/down)
    output = np.array(output)
    return output


def show_game(dqn_agent):

    new_board = board()
    new_board.display()
    state = create_state(new_board)
    state = state.reshape(1, dqn_agent.state_shape[1])

    turns = 0
    while not new_board.game_is_over():

        # get the next action
        output_array = dqn_agent.model.predict(state)[0]
        action = np.argmax(output_array)
        # if the action is invalid choose the next best
        while not new_board.is_valid_move(action):
            output_array[action] = -999999999999999
            action = np.argmax(output_array)

        if action == 0:
            move = "left"
        elif action == 1:
            move = "down"
        elif action == 2:
            move = "right"
        elif action == 3:
            move = "up"

        print(" turn: {} action: {} move: {} arrary: {} state: {}".format(turns, action, move, output_array, state))

        new_board.move_tiles(action)
        new_board.add_new_tile()
        new_board.display()
        state = create_state(new_board)
        state = state.reshape(1, dqn_agent.state_shape[1])
        turns += 1

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    episodes = 99999
    exploration_num = 750
    max_time = 0.5  # in seconds
    run(episodes, exploration_num, max_time)
    # pr.disable()
    # pr.print_stats()