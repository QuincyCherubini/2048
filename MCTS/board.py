import random
import numpy as np

class board:

    def __init__(self):

        self.score = 0
        self.tiles = [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]

        x_1 = random.getrandbits(2)
        y_1 = random.getrandbits(2)

        x_2 = random.getrandbits(2)
        y_2 = random.getrandbits(2)

        # make sure both pieces aren't the same
        while x_1 == x_2 and y_1 == y_2:
            x_2 = random.getrandbits(2)
            y_2 = random.getrandbits(2)

        self.tiles[x_1][y_1] = 2
        self.tiles[x_2][y_2] = 2

    def display(self):
        print("")
        print("score: {}".format(self.score))
        print("-----------------")
        for y in [0, 1, 2, 3]:
            print('', end='| ')
            for x in [0, 1, 2, 3]:
                val = self.tiles[x][y]
                print(val, end=' | ')
            print('')
            print("-----------------")

    def add_new_tile(self):

        x_1 = random.getrandbits(2)
        y_1 = random.getrandbits(2)
        random_val = random.getrandbits(11)

        val = 2
        # there is a ~10% chance of adding a new tile with the value of 4 instead of 2
        if random_val < 205:
            val = 4

        # if I have randomly selected a space with a tile already on it then check remaining tiles in order of likelyhood
        if self.tiles[x_1][y_1] != 0:

            if self.tiles[(x_1 + 3) % 4][(y_1 + 3) % 4] == 0:
                x_1 = (x_1 + 3) % 4
                y_1 = (y_1 + 3) % 4
            elif self.tiles[(x_1 + 3) % 4][(y_1 + 2) % 4] == 0:
                x_1 = (x_1 + 3) % 4
                y_1 = (y_1 + 2) % 4
            elif self.tiles[(x_1 + 3) % 4][(y_1 + 1) % 4] == 0:
                x_1 = (x_1 + 3) % 4
                y_1 = (y_1 + 1) % 4
            elif self.tiles[x_1][(y_1 + 2) % 4] == 0:
                y_1 = (y_1 + 2) % 4
            elif self.tiles[(x_1 + 1) % 4][(y_1 + 3) % 4] == 0:
                x_1 = (x_1 + 1) % 4
                y_1 = (y_1 + 3) % 4
            elif self.tiles[(x_1 + 2) % 4][(y_1 + 1) % 4] == 0:
                x_1 = (x_1 + 2) % 4
                y_1 = (y_1 + 1) % 4
            elif self.tiles[(x_1 + 2) % 4][(y_1 + 2) % 4] == 0:
                x_1 = (x_1 + 2) % 4
                y_1 = (y_1 + 2) % 4
            elif self.tiles[(x_1 + 1) % 4][(y_1 + 2) % 4] == 0:
                x_1 = (x_1 + 1) % 4
                y_1 = (y_1 + 2) % 4
            elif self.tiles[(x_1 + 2) % 4][(y_1 + 3) % 4] == 0:
                x_1 = (x_1 + 2) % 4
                y_1 = (y_1 + 3) % 4
            elif self.tiles[(x_1 + 1) % 4][(y_1 + 1) % 4] == 0:
                x_1 = (x_1 + 1) % 4
                y_1 = (y_1 + 1) % 4
            elif self.tiles[x_1][(y_1 + 1) % 4] == 0:
                y_1 = (y_1 + 1) % 4
            elif self.tiles[(x_1 + 2) % 4][y_1] == 0:
                x_1 = (x_1 + 2) % 4
            elif self.tiles[(x_1 + 1) % 4][y_1] == 0:
                x_1 = (x_1 + 1) % 4
            elif self.tiles[x_1][(y_1 + 3) % 4] == 0:
                y_1 = (y_1 + 3) % 4
            elif self.tiles[(x_1 + 3) % 4][y_1] == 0:
                x_1 = (x_1 + 3) % 4

        self.tiles[x_1][y_1] = val

    def move_tiles(self, output):

        has_merged = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        # left
        if output == 0:
            # each row is a self contained operation
            for y in [0, 1, 2, 3]:
                count_0 = 0
                if self.tiles[0][y] == 0:
                    count_0 += 1
                for x in [1, 2, 3]:
                    if self.tiles[x][y] != 0:
                        # if there have been 0's in the row
                        x_new = x - count_0
                        # set move the tile to it's new position
                        if count_0 > 0:
                            self.tiles[x_new][y] = self.tiles[x][y]
                            self.tiles[x][y] = 0
                        # check if the tile can be merged
                        if x_new != 0 and self.tiles[x_new - 1][y] == self.tiles[x_new][y] and has_merged[x_new - 1][y] == 0:
                            self.tiles[x_new - 1][y] += self.tiles[x_new][y]
                            self.tiles[x_new][y] = 0
                            self.score += self.tiles[x_new - 1][y]
                            has_merged[x_new - 1][y] = 1
                            count_0 += 1
                    else:
                        count_0 += 1

        # down
        elif output == 1:
            for x in [0, 1, 2, 3]:
                count_0 = 0
                if self.tiles[x][3] == 0:
                    count_0 += 1
                for y in [2, 1, 0]:
                    if self.tiles[x][y] != 0:
                        y_new = y + count_0
                        if count_0 > 0:
                            self.tiles[x][y_new] = self.tiles[x][y]
                            self.tiles[x][y] = 0
                        # check if the tile can be merged
                        if y_new != 3 and self.tiles[x][y_new] == self.tiles[x][y_new + 1] and has_merged[x][y_new + 1] == 0:
                            self.tiles[x][y_new + 1] += self.tiles[x][y_new]
                            self.tiles[x][y_new] = 0
                            self.score += self.tiles[x][y_new + 1]
                            has_merged[x][y_new + 1] = 1
                            count_0 += 1
                    else:
                        count_0 += 1

        # right
        elif output == 2:
            for y in [0, 1, 2, 3]:
                count_0 = 0
                if self.tiles[3][y] == 0:
                    count_0 += 1
                for x in [2, 1, 0]:
                    if self.tiles[x][y] != 0:
                        # if there have been 0's in the row
                        x_new = x + count_0
                        # set move the tile to it's new position
                        if count_0 > 0:
                            self.tiles[x_new][y] = self.tiles[x][y]
                            self.tiles[x][y] = 0
                        # check if the tile can be merged
                        if x_new != 3 and self.tiles[x_new + 1][y] == self.tiles[x_new][y] and has_merged[x_new + 1][y] == 0:
                            self.tiles[x_new + 1][y] += self.tiles[x_new][y]
                            self.tiles[x_new][y] = 0
                            self.score += self.tiles[x_new + 1][y]
                            has_merged[x_new + 1][y] = 1
                            count_0 += 1
                    else:
                        count_0 += 1

        # up
        elif output == 3:
            for x in [0, 1, 2, 3]:
                count_0 = 0
                if self.tiles[x][0] == 0:
                    count_0 += 1
                for y in [1, 2, 3]:
                    if self.tiles[x][y] != 0:
                        y_new = y - count_0
                        if count_0 > 0:
                            self.tiles[x][y_new] = self.tiles[x][y]
                            self.tiles[x][y] = 0
                        # check if the tile can be merged
                        if y_new != 0 and self.tiles[x][y_new] == self.tiles[x][y_new - 1] and has_merged[x][y_new - 1] == 0:
                            self.tiles[x][y_new - 1] += self.tiles[x][y_new]
                            self.tiles[x][y_new] = 0
                            self.score += self.tiles[x][y_new - 1]
                            has_merged[x][y_new - 1] = 1
                            count_0 += 1
                    else:
                        count_0 += 1

    def is_valid_move(self, output):

        # left
        if output == 0:
            for y in [0, 1, 2, 3]:
                if self.tiles[1][y] != 0:
                    if self.tiles[0][y] == 0 or self.tiles[1][y] == self.tiles[0][y] or self.tiles[1][y] == self.tiles[2][y]:
                        return True
                elif self.tiles[2][y] != 0 or self.tiles[3][y] != 0:
                    return True

                # if the 3rd and 4th row can merge
                if self.tiles[2][y] != 0:
                    if self.tiles[2][y] == self.tiles[3][y]:
                        return True
                # if the 4th row can move into the empty 3rd spot
                elif self.tiles[3][y] != 0:
                        return True

            return False

        # down
        elif output == 1:
            for x in [0, 1, 2, 3]:
                if self.tiles[x][2] != 0:
                    if self.tiles[x][3] == 0 or self.tiles[x][2] == self.tiles[x][3] or self.tiles[x][2] == self.tiles[x][1]:
                        return True
                elif self.tiles[x][1] != 0 or self.tiles[x][0] != 0:
                    return True

                # if it can merge with the top row
                if self.tiles[x][1] != 0:
                    if self.tiles[x][1] == self.tiles[x][0]:
                        return True
                elif self.tiles[x][0] != 0:
                        return True

            return False

        # right
        elif output == 2:
            for y in [0, 1, 2, 3]:
                if self.tiles[2][y] != 0:
                    if self.tiles[3][y] == 0 or self.tiles[2][y] == self.tiles[3][y] or self.tiles[2][y] == self.tiles[1][y]:
                        return True
                elif self.tiles[1][y] != 0 or self.tiles[0][y] != 0:
                    return True

                # if the pice in the first column can merge into the second
                if self.tiles[1][y] != 0:
                    if self.tiles[1][y] == self.tiles[0][y]:
                        return True
                elif self.tiles[0][y] != 0:
                    return True

            return False

        # up
        elif output == 3:
            for x in [0, 1, 2, 3]:
                if self.tiles[x][1] != 0:
                    if self.tiles[x][0] == 0 or self.tiles[x][1] == self.tiles[x][0] or self.tiles[x][1] == self.tiles[x][2]:
                        return True
                elif self.tiles[x][2] != 0 or self.tiles[x][3] != 0:
                    return True

                # if the bottom row can be merged with the 3rd row
                if self.tiles[x][2] != 0:
                    if self.tiles[x][2] == self.tiles[x][3]:
                        return True
                elif self.tiles[x][3] != 0:
                    return True

            return False

        # note: this line should never be called
        print("!!!!!!!!!!!!!! get move has failed")
        return False

    def game_is_over(self):

        # first check if we have any 0's
        # go in this order as it checks more likely spots first
        if self.tiles[0][0] == 0:
            return False
        if self.tiles[0][3] == 0:
            return False
        if self.tiles[3][0] == 0:
            return False
        if self.tiles[3][3] == 0:
            return False
        if self.tiles[0][1] == 0:
            return False
        if self.tiles[0][2] == 0:
            return False
        if self.tiles[3][1] == 0:
            return False
        if self.tiles[3][2] == 0:
            return False
        if self.tiles[1][0] == 0:
            return False
        if self.tiles[2][0] == 0:
            return False
        if self.tiles[1][3] == 0:
            return False
        if self.tiles[2][3] == 0:
            return False
        if self.tiles[1][1] == 0:
            return False
        if self.tiles[1][2] == 0:
            return False
        if self.tiles[2][1] == 0:
            return False
        if self.tiles[2][2] == 0:
            return False

        for y in [0, 1, 2, 3]:
            if self.tiles[1][y] == self.tiles[0][y] or self.tiles[1][y] == self.tiles[2][y] or self.tiles[2][y] == self.tiles[3][y]:
                return False

        for x in [0, 1, 2, 3]:
            if self.tiles[x][0] == self.tiles[x][1] or self.tiles[x][1] == self.tiles[x][2] or self.tiles[x][2] == self.tiles[x][3]:
                return False

        return True
