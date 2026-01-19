import numpy as np
import random
from scipy.ndimage import gaussian_filter, distance_transform_edt
import noise
import h5py
from PIL import Image
from map import MapVisualizer

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.plate_id = None
        self.crust_type = None
        self.is_boundary = False
        self.boundary_type = None  # convergent/divergent/transform
        self.height = 0.0
        self.tile_type = None
        self.color = (0,0,0)
        self.moisture = 0.0

class WorldGenerator:
    def __init__(self, width, height, plate_map, plates):
        self.WIDTH = width
        self.HEIGHT = height
        self.plate_map = plate_map
        self.plates = plates
        self.world = [[Cell(x, y) for x in range(self.WIDTH)] for y in range(self.HEIGHT)]

    # -----------------------------
    def create_base_world(self):
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                cell = self.world[y][x]
                cell.plate_id = self.plate_map[y][x]
                cell.crust_type = self.plates[cell.plate_id].crust_type

    # -----------------------------
    def detect_boundaries(self):
        for y in range(self.HEIGHT-1):
            for x in range(self.WIDTH-1):
                cell = self.world[y][x]
                neighbors = [self.world[y+1][x], self.world[y][x+1]]
                for neighbor in neighbors:
                    if cell.plate_id != neighbor.plate_id:
                        cell.is_boundary = True
                        neighbor.is_boundary = True

                        plate_a = self.plates[cell.plate_id]
                        plate_b = self.plates[neighbor.plate_id]
                        relative_velocity = plate_a.velocity - plate_b.velocity
                        normal = np.array([neighbor.x - cell.x, neighbor.y - cell.y])
                        normal = normal / (np.linalg.norm(normal) + 1e-6)
                        dot_product = np.dot(relative_velocity, normal)

                        if dot_product > 0.5:
                            b_type = "convergent"
                        elif dot_product < -0.5:
                            b_type = "divergent"
                        else:
                            b_type = "transform"

                        cell.boundary_type = b_type
                        neighbor.boundary_type = b_type

    # -----------------------------
    def apply_terrain(self):
        base_height = np.zeros((self.HEIGHT, self.WIDTH))
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                cell = self.world[y][x]
                if cell.crust_type == "continental":
                    base_height[y,x] = random.uniform(0, 500)
                else:
                    base_height[y,x] = random.uniform(-4000, -2000)

        # Горные границы
        mountain_mask = np.zeros((self.HEIGHT, self.WIDTH))
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                cell = self.world[y][x]
                if cell.is_boundary and cell.boundary_type=="convergent":
                    mountain_mask[y,x] = 1.0

        distance = distance_transform_edt(1 - mountain_mask)
        mountain_height = np.exp(-distance/5.0) * 4000
        base_height += mountain_height

        # Размытие
        height_array = gaussian_filter(base_height, sigma=1.5)

        # Локальный шум для неровностей
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                cell = self.world[y][x]
                scale = 20.0
                if cell.crust_type == "continental":
                    n = noise.pnoise2(x/scale, y/scale, octaves=3)
                    height_array[y,x] += n*50
                else:
                    n = noise.pnoise2(x/scale, y/scale, octaves=2)
                    height_array[y,x] += n*20

        # Присвоение клеткам
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                self.world[y][x].height = height_array[y,x]

    # -----------------------------
    def assign_biomes(self):
        ocean_mask = np.array([[1 if c.height >= 0 else 0 for c in row] for row in self.world])
        distance_to_water = distance_transform_edt(ocean_mask)
        scale_noise = 10.0

        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                cell = self.world[y][x]
                cell.moisture = np.exp(-distance_to_water[y,x]/30.0)
                lat = (y / self.HEIGHT) * 180 - 90
                temp = 1 - abs(lat)/90
                hum = cell.moisture

                # шум для границ
                n = noise.pnoise2(x/scale_noise, y/scale_noise, octaves=2)
                temp += n*0.1
                hum += n*0.1

                if cell.height > 2500:
                    cell.tile_type = "mountain"
                    cell.color = (120,110,100)
                elif cell.height < -2000:
                    cell.tile_type = "deep_ocean"
                    cell.color = (0,0,128)
                elif cell.height < 0:
                    cell.tile_type = "shelf"
                    cell.color = (0,0,255)
                else:
                    if temp > 0.6:
                        if hum > 0.5:
                            cell.tile_type = "tropical_forest"
                            cell.color = (34,139,34)
                        else:
                            cell.tile_type = "desert"
                            cell.color = (237,201,175)
                    elif temp > 0.3:
                        if hum > 0.5:
                            cell.tile_type = "temperate_forest"
                            cell.color = (107,142,35)
                        else:
                            cell.tile_type = "grassland"
                            cell.color = (189,183,107)
                    else:
                        if temp > 0.1:
                            cell.tile_type = "tundra"
                            cell.color = (198,226,255)
                        else:
                            cell.tile_type = "ice"
                            cell.color = (255,255,255)

    # -----------------------------
    # Экспорт HDF5
    # -----------------------------
    def export_hdf5(self, filename="world_data.h5"):
        height_array = np.array([[c.height for c in row] for row in self.world], dtype=np.float32)
        biome_array = np.array([[c.tile_type for c in row] for row in self.world], dtype='S')
        crust_array = np.array([[c.crust_type for c in row] for row in self.world], dtype='S')
        moisture_array = np.array([[c.moisture for c in row] for row in self.world], dtype=np.float32)

        with h5py.File(filename, "w") as f:
            f.create_dataset("height", data=height_array)
            f.create_dataset("biome", data=biome_array)
            f.create_dataset("crust_type", data=crust_array)
            f.create_dataset("moisture", data=moisture_array)
        print(f"[INFO] World data exported to {filename}")

    # -----------------------------
    # Экспорт PNG с градиентами биомов
    # -----------------------------
    def export_png(self, filename="world_map_gradient.png", steps=3):
        img = MapVisualizer.blend_biomes(self.world, steps=steps)
        Image.fromarray(img).save(filename)
        print(f"[INFO] World PNG exported to {filename}")

    # -----------------------------
    # Экспорт PNG карты высот
    # -----------------------------
    def export_heightmap_png(self, filename="heightmap.png"):
        height_array = np.array([[c.height for c in row] for row in self.world], dtype=np.float32)
        min_h, max_h = height_array.min(), height_array.max()
        norm = ((height_array - min_h)/(max_h - min_h) * 255).astype(np.uint8)
        im = Image.fromarray(norm).convert("L")
        im.save(filename)
        print(f"[INFO] Heightmap PNG exported to {filename}")
