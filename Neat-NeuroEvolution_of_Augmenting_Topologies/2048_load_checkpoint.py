from board import board
import os
import neat
import numpy as np
import pickle
import time
import math


def display_winner(genome, genome_config):

    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    test_board = board()
    test_board.display()

    # load initial state
    observation = create_input(test_board)

    for t in range(2000):

        action = net.activate(observation)
        action_rounded = np.argmax(action)

        test_count = 0
        while not test_board.is_valid_move(action_rounded) and test_count <= 5:
            # print("t: {} action_rounded: {} action: {}".format(t, action_rounded, action))
            action[action_rounded] = 0
            action_rounded = np.argmax(action)
            test_count += 1
        if test_count >= 5:
            break

        if test_board.is_valid_move(action_rounded):
            test_board.move_tiles(action_rounded)
            test_board.add_new_tile()

            if action_rounded == 0:
                move = "left"
            elif action_rounded == 1:
                move = "down"
            elif action_rounded == 2:
                move = "right"
            elif action_rounded == 3:
                move = "up"

            action = net.activate(observation)
            print("t: {} action: {} move: {}".format(t, action, move))
            test_board.display()

        observation = create_input(test_board)

        # if test_board.game_is_over():
        #     print("Episode finished after {} timesteps with score: {}".format(t+1, test_board.score))
        #     break

def run(best_path, config_file):

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Checkpointer.restore_checkpoint(best_path)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100))

    winner = p.run(eval_genomes, 1)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    display_winner(winner, config)

    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)

def eval_genomes(genomes, genome_config):

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
        total_reward = 0.0

        test_board = board()

        # load initial state
        observation = create_input(test_board)

        for t in range(2000):

            action = net.activate(observation)
            action_rounded = np.argmax(action)

            test_count = 0
            while not test_board.is_valid_move(action_rounded) and test_count <= 5:
                action[action_rounded] = 0
                action_rounded = np.argmax(action)
                test_count += 1
            if test_count >= 5:
                break

            if test_board.is_valid_move(action_rounded):
                test_board.move_tiles(action_rounded)
                test_board.add_new_tile()

            observation = create_input(test_board)

            if test_board.game_is_over():
                # print("Episode finished after {} timesteps with score: {}".format(t+1, total_reward))
                break

        total_reward = test_board.score
        genome.fitness = total_reward

def create_input(board):

    output = []

    # give the log base 2 of the board
    for x in range(0, 4):
        for y in range(0, 4):
            if board.tiles[x][y] == 0:
                output.append(0)
            else:
                output.append(math.log(board.tiles[x][y], 2))

    # 1 if the tile is 0, 0 other wise
    for x in range(0, 4):
        for y in range(0, 4):
            if board.tiles[x][y] == 0:
                output.append(1)
            else:
                output.append(0)

    # this is the bias node
    output.append(1)

    return output


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-2048')
    best_path = os.path.join(local_dir, 'neat-checkpoint-354')
    run(best_path, config_path)
