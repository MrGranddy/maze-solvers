import argparse
from typing import List, Tuple, Optional

import random

import numpy as np
from PIL import Image

# Maze coding
# 0: empty cell
# 1: wall
# 2: visited (used in generation)
# 3: exitable
# 4: start


def _create_empty_maze(width: int, height: int) -> List[List[int]]:
    """This function generates an empty maze in the form of [[Int]], the given
    width and height is doubled and incremened to add walls in-between cells,
    in the and given dimensions describe the walkable cells.

    Args:
        width (int): Number of empty cells in the horizontal direction.
        height (int): Number of empty cells in the vertical direction.

    Returns:
        List[List[int]]: The empty maze in the form of [[Int]]. The maze is represented
        as a 2D array of integers, where 0 represents an empty cell, 1 represents a wall.
    """

    _maze = [[0 for _ in range(2 * width + 1)] for _ in range(2 * height + 1)]

    for x in range(2 * width + 1):
        for y in range(2 * height + 1):
            # Add walls in-between cells
            if x % 2 == 0 or y % 2 == 0:
                _maze[y][x] = 1

    return _maze


def _check_visited(_maze: List[List[int]], x: int, y: int) -> bool:
    """This function checks if a given cell in a maze is visited. The given coordinates
    are empty cell coordinates, the function converts them to maze coordinates.

    Args:
        _maze (List[List[int]]): The maze in the form of [[Int]].
        x (int): The x coordinate of the cell. (horizontal)
        y (int): The y coordinate of the cell. (vertical)

    Returns:
        bool: True if the cell is visited, False otherwise.
    """

    return _maze[2 * y + 1][2 * x + 1] == 2


def _get_unvisited_neighbours(
    _maze: List[List[int]], x: int, y: int, width: int, height: int
) -> List[Tuple[int, int]]:
    """This function returns the unvisited neighbours of a given cell in a maze.

    Args:
        _maze (List[List[int]]): The maze in the form of [[Int]].
        x (int): The x coordinate of the cell. (horizontal)
        y (int): The y coordinate of the cell. (vertical)
        width (int): Number of empty cells in the horizontal direction.
        height (int): Number of empty cells in the vertical direction.

    Returns:
        List[Tuple[int, int]]: _description_
    """

    # Get the coordinates of the neighbours
    virtual_neig_coords: List[Tuple[int, int]] = [
        (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
    ]

    # Filter out out of bounds cells
    real_neig_coords: List[Tuple[int, int]] = list(
        filter(
            lambda c: (0 <= c[0] <= width - 1) and (0 <= c[1] <= height - 1),
            virtual_neig_coords,
        )
    )

    # Filter out visited cells
    valid_neig_coords: List[Tuple[int, int]] = list(
        filter(lambda c: not _check_visited(_maze, c[0], c[1]), real_neig_coords)
    )

    return valid_neig_coords


def _get_exitable_walls(
    _maze: List[List[int]], width: int, height: int
) -> List[Tuple[int, int]]:
    """This function returns the exitable cells of a maze.

    Args:
        _maze (List[List[int]]): The maze in the form of [[Int]].
        width (int): Number of empty cells in the horizontal direction.
        height (int): Number of empty cells in the vertical direction.

    Returns:
        List[Tuple[int, int]]: The exitable cells of a maze.
    """

    exitable_coords = [(y, 0) for y in range(2 * height + 1) if _maze[y][1] == 0]
    exitable_coords += [
        (y, 2 * width) for y in range(2 * height + 1) if _maze[y][2 * width - 1] == 0
    ]
    exitable_coords += [(0, x) for x in range(2 * width + 1) if _maze[1][x] == 0]
    exitable_coords += [
        (2 * height, x) for x in range(2 * width + 1) if _maze[2 * height - 1][x] == 0
    ]

    return exitable_coords


class MazeGenerator:
    """This class generates a maze in the form of [[Int]] with each call to the __call__"""

    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        """This function initializes the maze generator. The given width and height is
        doubled and incremened to add walls in-between cells, in the and given
        dimensions describe the walkable cells.

        Args:
            width (int): Number of empty cells in the horizontal direction.
            height (int): Number of empty cells in the vertical direction.
            seed (Optional[int], optional): The seed for the random number generator.
            Defaults to None.
        """

        self.width = width
        self.height = height

        if seed:
            random.seed(seed)

    def _generate_maze(self):
        """This function generates a maze in the form of [[Int]]. The maze is represented as a
        2D array of integers, where 0 represents an empty cell, 1 represents a wall.

        Returns:
            List[List[int]]: The maze in the form of [[Int]].
        """

        # Generate an empty maze
        _maze = _create_empty_maze(self.width, self.height)

        # Genearation stack
        stack: List[Tuple[int, int]] = []

        # Pick a random cell to start
        curr: Tuple[int, int] = (
            random.randint(0, self.width - 1),
            random.randint(0, self.height - 1),
        )

        # Mark the cell as visited and add it to the stack
        _maze[curr[1] * 2 + 1][curr[0] * 2 + 1] = 2
        stack.append(curr)

        # While stack is not empty
        while len(stack) > 0:

            # Pop cell and make current
            curr = stack.pop(-1)

            # Get unvisited neighbours
            neighs = _get_unvisited_neighbours(_maze, *curr, self.width, self.height)

            # Choose one of the neighbours if exists, else skip iteration
            if len(neighs) > 0:
                sel = random.choice(neighs)
                # We also add the current back to the stack
                stack.append(curr)
            else:
                continue

            # Remove the wall between current and the selected cell
            _maze[(sel[1] + curr[1]) + 1][(sel[0] + curr[0]) + 1] = 0

            # Mark choosen cell as visited
            _maze[sel[1] * 2 + 1][sel[0] * 2 + 1] = 2

            # Push selected cell to stack
            stack.append(sel)

        # Convert visited markers into empty markers
        for i, _ in enumerate(_maze):
            for j, _ in enumerate(_maze[0]):
                if _maze[i][j] == 2:
                    _maze[i][j] = 0

        # Add START and EXIT markers
        exitable_coords = _get_exitable_walls(_maze, self.width, self.height)
        enter_coord, exit_coord = random.sample(exitable_coords, 2)
        _maze[enter_coord[0]][enter_coord[1]] = 3  # Mark as START
        _maze[exit_coord[0]][exit_coord[1]] = 4  # Mark as EXIT

        return _maze

    def __call__(self) -> str:
        """This function generates a maze in the form of [[Int]] and returns it as a string.
        '#' represents a wall, ' ' represents an empty cell, 'S' represents the start cell
        and 'E' represents the exit cell.

        Returns:
            str: The maze in the form of a string.
        """

        _maze: List[List[int]] = self._generate_maze()

        maze_str_list: List[str] = []
        for _row in _maze:
            row_str_list: List[str] = []
            for _cell in _row:
                if _cell == 1:
                    row_str_list.append("#")
                elif _cell == 0:
                    row_str_list.append(" ")
                elif _cell == 3:
                    row_str_list.append("S")
                elif _cell == 4:
                    row_str_list.append("E")
            maze_str_list.append("".join(row_str_list))

        return "\n".join(maze_str_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Maze Generator")
    parser.add_argument("width", type=int, help="Width of the maze")
    parser.add_argument("height", type=int, help="Height of the maze")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    MazeGen = MazeGenerator(args.width, args.height)
    maze = MazeGen()

    if args.output:

        if args.output.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):

            grid: List[List[List[int]]] = []

            for row in maze.splitlines():
                grid_row: List[List[int]] = []
                for col in row:
                    if col == "#":
                        grid_row.append([0, 0, 0])
                    elif col == " ":
                        grid_row.append([255, 255, 255])
                    elif col == "S":
                        grid_row.append([0, 255, 0])
                    elif col == "E":
                        grid_row.append([255, 0, 0])
                grid.append(grid_row)

            grid_darray = np.array(grid, dtype=np.uint8)

            grid_image = Image.fromarray(grid_darray).resize(
                (int(grid_darray.shape[1] * 10), int(grid_darray.shape[0] * 10)),
                resample=Image.Resampling.NEAREST,
            )

            grid_image.save(args.output)

        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(maze)
    else:
        print(maze)
