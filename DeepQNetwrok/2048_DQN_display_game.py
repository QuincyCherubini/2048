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

class DQN:
    # my board is the environment
    def __init__(self):
        self.memory = deque(maxlen=5000)
        self.state_shape = [1, 160]
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.002
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.005
        self.tau = 0.125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(85, input_dim=self.state_shape[1]))
        model.add(Dense(4))  # Action space for 2048
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

def run(model):

    dqn_agent = DQN()
    dqn_agent.model = model
    show_game(dqn_agent)

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

        print(" turn: {} action: {} move: {} arrary: {}".format(turns, action, move, output_array))

        new_board.move_tiles(action)
        new_board.add_new_tile()
        new_board.display()
        state = create_state(new_board)
        state = state.reshape(1, dqn_agent.state_shape[1])
        turns += 1

if __name__ == "__main__":
    model = keras.models.load_model("./obj/10/trial-3002--3486.model")
    run(model)
