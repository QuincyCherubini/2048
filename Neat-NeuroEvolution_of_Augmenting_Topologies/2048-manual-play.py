from board import board

def run(test_board):

    test_board.display()

    while not test_board.game_is_over():

        # get the user input
        while True:
            val = input("Enter your value: ")

            if val == "a" or val == "s" or val == "d" or val == "w":
                if val == "a":
                    output = 0
                elif val == "s":
                    output = 1
                elif val == "d":
                    output = 2
                elif val == "w":
                    output = 3
                break

        # move the piece, add a new one, display the new board
        if test_board.is_valid_move(output):
            test_board.move_tiles(output)
            test_board.add_new_tile()
            test_board.display()


if __name__ == "__main__":
    test_board = board()
    run(test_board)
