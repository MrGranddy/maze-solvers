# maze-solvers

A collection of maze solving algorithms implemented in Python.

## Maze Generation Script

The `maze_generator.py` script generates a maze of a given size, inside it contains MazeGenerator class which can be used to generate a maze of a given size.
You can either use the script with arguments or import the class and use it in code. Console arguments are as follows:

- `width` and `height` are the dimensions of the maze given in the respective order.
- `--seed` is the seed for the random number generator, if not given a random seed is used.
- `--output` is the output file, if not given the maze is printed to the console, if given with an image extension the maze is saved as an image, otherwise the maze is saved as a text file.