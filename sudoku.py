import numpy as np
import copy
import random

def to_set(*lists):
    """

    :param lst:
    :return: set of all elements contained in the lists
    """
    s = set()
    for lst in lists:
        s = s.union({i for i in np.array(lst).flatten()})
    return s
#todo: add Grid class
class Grid():
    # , grid=self.grid
    pass

class Sudoku(Grid):
    size = (9, 9)

    def __init__(self, grid=None, size=size):
        """

        :param grid:
        :param size: tuple
        """
        if type(size) == tuple and size[0] == size[1]:
            self.size = size
        self.possible_numbers = to_set(range(1, self.size[0] + 1))
        if grid is not None and self.valid_shape(grid):
            self.grid = grid
        else:
            self.grid = self.generate_grid()

    def generate_grid(self):
        n_rows, n_cols = self.size
        grid = np.zeros(self.size, dtype=int)
        filled = 0
        stack = []
        row, col=0,0
        while filled<n_cols*n_rows:
            pos = self.possible_numbers-to_set(self.row(row, grid),self.col(col,grid),self.square(row, col,grid))
            while len(pos)<=0:
                grid, row, col, pos, filled=stack.pop()

            if len(pos) ==1:
                grid[row][col] = pos.pop()
            else:
                rand_num=random.sample(pos,1)[0]#.pop()
                pos.remove(rand_num)
                stack.append((copy.copy(grid), row, col, pos, filled))
                grid[row][col] = rand_num
            filled+=1
            col = (col+1) % n_cols
            row = row+1 if col==0 else row
        self.print_grid(grid)
        self.grid=grid
        print(self.is_valid())
        return grid
                            

    def possible_numbers(self, row, col):
        nums = {}

    def row(self, row, grid=None):
        grid=self.grid if grid is None else grid
        return grid[row, :]

    def col(self, col, grid=None):
        grid=self.grid if grid is None else grid
        return grid[:, col]

    def print_sudoku(self, grid=None):
        if grid is not None:
            self.print_grid(grid)
        else:
            self.print_grid(self.grid)

    def print_grid(self, grid):
        n_rows, n_cols = self.size
        space = int(np.sqrt(n_rows))
        if self.valid_shape(grid):
            grid = np.reshape(grid, (n_rows, n_cols))
            for row in range(n_rows):
                row_str = ''
                for col in range(n_cols):
                    num = str(grid[row, col]) if grid[row, col] != 0 else '-'
                    row_str += num + ' ' if col % space != 0 else '| ' + num + ' '
                if row % space == 0:
                    print('-' * (2 * (space + n_rows + 1) - 1))
                print(row_str + '|')

            print('-' * (2 * (space + n_rows + 1) - 1))

    def guess(self, row, col, num):
        self.grid[row, col] = num
        if self.row_is_valid(row) and self.col_is_valid(col) and self.square_is_valid(row, col):
            return True
        else:
            return False

    def is_valid(self):
        n_rows, n_cols = self.size
        rows = all([self.row_is_valid(row) for row in range(n_rows)])
        cols = all([self.col_is_valid(col) for col in range(n_cols)])
        squares = [(row * rows, col * cols) for row in range(int(np.sqrt(rows))) for col in range(int(np.sqrt(cols)))]
        squares_valid = all([self.square_is_valid(row, col) for row, col in squares])
        return all([rows, cols, squares_valid])

    def row_is_valid(self, row, grid=None):
        grid=self.grid if grid is None else grid
        return all(np.bincount(self.row(row, grid), minlength=self.size[0] + 1)[1:] <= np.ones(self.size[0]))

    def col_is_valid(self, col, grid=None):
        grid=self.grid if grid is None else grid
        return all(np.bincount(self.col(col,grid), minlength=self.size[0] + 1)[1:] <= np.ones(self.size[0]))

    def square_is_valid(self, row, col, grid=None):
        grid=self.grid if grid is None else grid
        return all(
            np.bincount(self.square(row, col,grid).flatten(), minlength=self.size[0] + 1)[1:] <= np.ones(self.size[0]))

    def square(self, row, col, grid=None):
        grid=self.grid if grid is None else grid
        n_rows, n_cols = np.sqrt(self.size)
        n_rows, n_cols = int(n_rows), int(n_cols)
        return grid[n_rows * (row // n_rows):n_rows * (row // n_rows + 1),
               n_cols * (col // n_cols):n_cols * (col // n_cols + 1)]

    def valid_shape(self, grid):
        try:
            np.reshape(grid, self.size)
            # for i in grid.flatten():
            #     if type(i) != int:
            #         return False
            return True
        except:
            raise ValueError("Grid must be of the size %s" % (self.size,))
            return False


def main():
    sudoku = Sudoku(size=(9, 9))
    # print(sudoku.size, sudoku.possible_numbers)
    # print(sudoku.is_valid())
    # sudoku.print_sudoku()


if __name__ == "__main__":
    main()
