from typing import List, Tuple

def _get_neighbours(maze: List[List[str]], x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    """This function returns the neighbours of a cell in a maze.

    Args:
        maze (List[List[str]]): The maze in the form of [[str]].
        x (int): The x coordinate of the cell. (horizontal)
        y (int): The y coordinate of the cell. (vertical)
        width (int): Number of cells in the horizontal direction.
        height (int): Number of cells in the vertical direction.

    Returns:
        List[Tuple[int, int]]: The neighbours of a cell in a maze.
        """

    neighbours = [ (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] if maze[y + dy][x + dx] == " " and 0 <= x + dx <= width - 1 and 0 <= y + dy <= height - 1]

    return neighbours

class DeadendSolver:
    def __init__(self):
        pass

    def __call__(self, maze: str) -> List[Tuple[int, int]]:

        maze_matrix = [list(row) for row in maze.strip().splitlines()]
        width = len(maze_matrix[0])
        height = len(maze_matrix)

        start_cell = None
        end_cell = None

        for x in range(width):
            for y in range(height):
                if maze_matrix[y][x] == "S":
                    start_cell = (x, y)
                elif maze_matrix[y][x] == "E":
                    end_cell = (x, y)
        
        maze_matrix[start_cell[1]][start_cell[0]] = " "
        maze_matrix[end_cell[1]][end_cell[0]] = " "

        deadends = set( (x, y, neighs[0]) for y in range(1, height - 1) for x in range(1, width - 1) for neighs in [_get_neighbours(maze_matrix, x, y, width, height)] if maze_matrix[y][x] == " " and len(neighs) == 1 )

        import numpy as np
        from PIL import Image
            
        cnt = 0
        while deadends:
            cnt += 1
            x, y, neighbour = deadends.pop()

            # Block the deadend
            maze_matrix[y][x] = "#"

            # Remove the cell from deadends
            deadends.discard( (x, y, neighbour) )

            # Neighbours of the neighbour
            neig_neig = _get_neighbours(maze_matrix, *neighbour, width, height)

            # If the neighbour has only one neighbour, add it to deadends
            if len(neig_neig) == 1:
                deadends.add( (neighbour[0], neighbour[1], neig_neig[0]) )


            if cnt % 100 == 0:

                maze_array = np.array( [ [ [255, 255, 255] if c == " " else [0, 0, 0] for c in row ] for row in maze_matrix ], dtype=np.uint8 )
                img = Image.fromarray(maze_array).resize((width * 10, height * 10), resample=Image.Resampling.NEAREST)
                img.save("maze_%s.png" % str(cnt).zfill(6))
    
        maze_array = np.array( [ [ [255, 255, 255] if c == " " else [0, 0, 0] for c in row ] for row in maze_matrix ], dtype=np.uint8 )
        img = Image.fromarray(maze_array).resize((width * 10, height * 10), resample=Image.Resampling.NEAREST)
        img.save("maze_FINAL.png")

        return []