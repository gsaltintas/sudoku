import copy
import random

import numpy as np


# todo: add docstring
def to_set(*lists):
    """

    :param lst:
    :return: set of all elements contained in the lists
    """
    s = set()
    for lst in lists:
        s = s.union({i for i in np.array(lst).flatten()})
    return s


def find_min_key(d: dict):
    min_key = list(d.keys())[0]
    min_size = len(d[min_key])
    for key in d.keys():
        if len(d[key]) < min_size:
            min_key = key
            min_size = len(d[key])
    return min_key


# todo: add Grid class
class Grid():
    # , grid=self.grid
    def __init__(self, array):
        pass


class SudokuSolver():
    def generate_possible_numbers(self, sudoku):
        pos = {}
        n_rows, n_cols = sudoku.size
        for row in range(n_rows):
            for col in range(n_cols):
                if sudoku[row, col] == 0:
                    pos[row, col] = sudoku.get_possible_numbers(row, col)
        return pos

    def find_min_entry(self, d):
        return find_min_key(d)

    def update_possibilities(self, sudoku, possibilities, ind, guess):
        n_rows, n_cols = sudoku.size
        nr, nc = int(np.sqrt(n_rows)), int(np.sqrt((n_cols)))
        row, col = ind
        row_ind = {(row, c) for c in range(n_cols)}
        col_ind = {(r, col) for r in range(n_rows)}
        square_ind = {(nr * (row // nr) + i, nc * (col // nc) + j) for i in range(nr)
                      for j in range(nc)}
        indices = row_ind.union(col_ind.union(square_ind)) - {ind}
        for (r, c) in indices:
            if (r, c) in possibilities.keys():
                possibilities[(r, c)] = possibilities[(r, c)] - {guess}
                if len(possibilities[(r, c)]) == 0:
                    return False
        return True

    def solve(self, sudoku):
        n_rows, n_cols = sudoku.size
        solved = Sudoku(grid=sudoku.grid, size=sudoku.size)
        possibilities = self.generate_possible_numbers(solved)
        filled = solved.count_filled()
        stack = []
        popped = 0
        while filled < n_cols * n_rows:
            while len(possibilities) == 0:
                solved.grid, row, col, possibilities, filled = stack.pop()

            row, col = self.find_min_entry(possibilities)
            pos = possibilities[(row, col)]

            while len(pos) <= 0:
                solved.grid, row, col, possibilities, filled = stack.pop()
                row, col = self.find_min_entry(possibilities)
                pos = possibilities[(row, col)]

            if len(pos) == 1:
                num = pos.pop()
                possibilities.pop((row, col))
            else:
                num = random.sample(pos, 1)[0]  # .pop()
                pos.remove(num)
                stack.append((copy.copy(solved.grid), row, col, copy.copy(possibilities), filled))
                possibilities.pop((row, col))
            if solved.guess(row, col, num):
                solved[row, col] = num
                filled = solved.count_filled()
                if not self.update_possibilities(solved, possibilities, (row, col), num):
                    solved.grid, row, col, possibilities, filled = stack.pop()


            else:
                popped += 1
                solved.grid, row, col, possibilities, filled = stack.pop()
            filled = solved.count_filled()
        return solved


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
        row, col = 0, 0
        while filled < n_cols * n_rows:
            pos = self.possible_numbers - to_set(self.row(row, grid), self.col(col, grid), self.square(row, col, grid))
            while len(pos) <= 0:
                grid, row, col, pos, filled = stack.pop()

            if len(pos) == 1:
                grid[row][col] = pos.pop()
            else:
                rand_num = random.sample(pos, 1)[0]  # .pop()
                pos.remove(rand_num)
                stack.append((copy.copy(grid), row, col, pos, filled))
                grid[row][col] = rand_num
            filled += 1
            col = (col + 1) % n_cols
            row = row + 1 if col == 0 else row
        # self.print_grid(grid)
        self.grid = grid
        # print(self.is_valid())
        return grid

    def remove(self, remaining_nos):
        n_rows, n_cols = self.size[0], self.size[1]
        pops = n_rows * n_cols - remaining_nos
        pop_indices = random.sample([(row, col) for row in range(n_rows) for col in range(n_cols)], pops)
        for ind in pop_indices:
            self[ind] = 0

    def solve(self):
        pass

    def get_possible_numbers(self, row, col):
        return self.possible_numbers - to_set(self.row(row), self.col(col), self.square(row, col))

    def row(self, row, grid=None):
        grid = self.grid if grid is None else grid
        return grid[row, :]

    def col(self, col, grid=None):
        grid = self.grid if grid is None else grid
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
            self.grid[row, col] = 0
            return False

    def is_valid(self):
        n_rows, n_cols = self.size
        rows = all([self.row_is_valid(row) for row in range(n_rows)])
        cols = all([self.col_is_valid(col) for col in range(n_cols)])
        squares = [(row * rows, col * cols) for row in range(int(np.sqrt(rows))) for col in range(int(np.sqrt(cols)))]
        squares_valid = all([self.square_is_valid(row, col) for row, col in squares])
        return all([rows, cols, squares_valid])

    def count_filled(self):
        return sum(np.bincount(self.grid.flatten(), minlength=2)[1:])

    def row_is_valid(self, row, grid=None):
        grid = self.grid if grid is None else grid
        return all(np.bincount(self.row(row, grid), minlength=self.size[0] + 1)[1:] <= np.ones(self.size[0]))

    def col_is_valid(self, col, grid=None):
        grid = self.grid if grid is None else grid
        return all(np.bincount(self.col(col, grid), minlength=self.size[0] + 1)[1:] <= np.ones(self.size[0]))

    def square_is_valid(self, row, col, grid=None):
        grid = self.grid if grid is None else grid
        return all(
            np.bincount(self.square(row, col, grid).flatten(), minlength=self.size[0] + 1)[1:] <= np.ones(self.size[0]))

    def square(self, row, col, grid=None):
        grid = self.grid if grid is None else grid
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
            print(ValueError("Grid must be of the size %s" % (self.size,)))
            return False

    def __getitem__(self, key):
        if type(key) is not tuple or len(key) != 2 or (type(key[0]) is not int and type(key[1]) is not int):
            raise ValueError("Key must be a tuple of 2 integers.")
        else:
            return self.grid[key[0]][key[1]]

    def __setitem__(self, key, value):
        if type(key) is not tuple or len(key) != 2 or (type(key[0]) is not int and type(key[1]) is not int):
            print(TypeError("Key must be a tuple of 2 integers."))
        elif value not in self.possible_numbers and value != 0:
            print(ValueError("Invalid number, numbers should be in %s" % self.possible_numbers))
        elif not self.guess(key[0], key[1], value):
            print(ValueError("Invalid guess for (%d, %d)" % (key[0], key[1])))
        else:
            self.grid[key[0]][key[1]] = value

    def __eq__(self, sudoku):
        if type(sudoku) is not Sudoku:
            print(TypeError("Value must be of type Sudoku."))
        else:
            return (self.grid == sudoku.grid).all()


def main():
    sudoku = Sudoku(size=(9, 9))
    # sudoku.print_sudoku()
    removed = Sudoku(sudoku.grid, size=sudoku.size)
    removed.remove((sudoku.size[0] * sudoku.size[1]) // 3 * 2)
    # sudoku.print_sudoku()
    solver = SudokuSolver()
    solved = solver.solve(removed)
    solved.print_sudoku()
    print(solved == sudoku)


if __name__ == "__main__":
    main()
