# Sudoku Application in Python
This is a simple sudoku application consisting of a [sudoku generator](sudoku.py) and sudoku solver 
from a sudoku lover.
- Sudoku class allows you to create different sized sudokus and check for validity.

## Implementation
### Generating Sudoku
To generate a Sudoku *s* of size (*n, n*), Starting from (*row,col*)=(*0,0*):
1. Get *pos*: the set of possible numbers for *s[row,col]*.
2. While *pos* is empty, return to the previous state of generator.
3. If *pos* has a single entry assign it to *s[row, col]*, otherwise pick one and assign it to *s[row,col]* and save the state(grid before assignment, row, col, other entries in pos, number of filled entries in the grid).
4. Repeat [*1-3*] for all indices (*row, col*) in sudoku grid.

### Solving Sudoku
Given a Sudoku of size (*n,n*) with *m* empty slots, SudokuSolver 
1. Identify possible numbers, *possibilities*, for each (*r,c*) such that sudoku[r, c] is empty.
2. Find the index (*r,c*) with the least number of possible numbers (if there are multiple such (r, c) return the first one).
3. If *possibilities[r, c]* has a single entry pick it as *num* otherwise,
 pick one of them as *num* and add the last state of the solver to the stack .
4. Try assigning *num* to [*r,c*]. If the guess is not valid return to the previous state of the solver.
5. Repeat [*2-4*] until all entries are filled in the sudoku grid. 