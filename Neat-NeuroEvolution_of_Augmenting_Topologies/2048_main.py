from board import board
import neat
import numpy as np
import os
import math

def eval_genomes(genomes, genome_config):

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
        total_reward = 0.0

        test_board = board()

        # load initial state
        observation = create_input(test_board)

        for t in range(1000):

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
            else:
                print("This should never happen")

            observation = create_input(test_board)

            if test_board.game_is_over():
                # print("Episode finished after {} timesteps with score: {}".format(t+1, total_reward))
                break

        total_reward = test_board.score
        genome.fitness = total_reward

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


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

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-2048')
    run(config_path)
