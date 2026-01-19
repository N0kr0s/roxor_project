import numpy as np
import matplotlib.pyplot as plt

class MapVisualizer:
    @staticmethod
    def blend_biomes(world, steps=3):
        HEIGHT = len(world)
        WIDTH = len(world[0])
        base_colors = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        heights = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

        for y in range(HEIGHT):
            for x in range(WIDTH):
                base_colors[y, x] = np.array(world[y][x].color, dtype=np.float32)
                heights[y, x] = world[y][x].height

        img = base_colors.copy()

        for step in range(1, steps + 1):
            temp = img.copy()
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    cell = world[y][x]
                    neighbors = []
                    weights = []

                    for dy in range(-step, step + 1):
                        for dx in range(-step, step + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < HEIGHT and 0 <= nx < WIDTH and (ny != y or nx != x):
                                neighbor_cell = world[ny][nx]
                                # Вода и суша не смешиваем
                                if (cell.tile_type in ["deep_ocean","shelf"] and neighbor_cell.tile_type not in ["deep_ocean","shelf"]):
                                    continue
                                if (cell.tile_type not in ["deep_ocean","shelf"] and neighbor_cell.tile_type in ["deep_ocean","shelf"]):
                                    continue
                                neighbors.append(neighbor_cell.color)
                                weights.append(heights[ny,nx]+1e-3)

                    if neighbors:
                        neighbors = np.array(neighbors)
                        weights = np.array(weights)
                        weights /= (weights.sum()+1e-6)
                        avg = np.sum(neighbors*weights[:,None], axis=0)
                        factor = 0.5/step
                        temp[y,x] = (1-factor)*img[y,x] + factor*avg

            img = temp

        return np.clip(img,0,255).astype(np.uint8)

    @staticmethod
    def visualize(world, steps=3, filename=None):
        img = MapVisualizer.blend_biomes(world, steps=steps)
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.axis("off")
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
