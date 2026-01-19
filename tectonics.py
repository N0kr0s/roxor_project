import numpy as np
import random
import heapq
from noise import pnoise2

class TectonicPlate:
    def __init__(self, plate_id):
        self.id = plate_id

        # Тип коры
        self.crust_type = random.choices(
            ["continental", "oceanic", "mixed"],
            weights=[0.35, 0.4, 0.25]
        )[0]

        # Толщина и плотность
        if self.crust_type == "continental":
            self.thickness_km = random.uniform(120, 200)
            self.density = 2.7
        elif self.crust_type == "oceanic":
            self.thickness_km = random.uniform(15, 40)
            self.density = 3.0
        else:
            self.thickness_km = random.uniform(60, 140)
            self.density = random.uniform(2.7, 3.0)

        # Скорость и направление
        speed_cm_year = random.uniform(1, 10)
        angle = random.uniform(0, 2 * np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed_cm_year

        # Возраст океанической коры
        if self.crust_type == "oceanic":
            self.oceanic_age_myr = random.uniform(0, 180)
        else:
            self.oceanic_age_myr = None

class TectonicsGenerator:
    def __init__(self, width=200, height=200, plates_count=5, noise_scale=0.04, noise_strength=500.0, seed=None):
        self.WIDTH = width
        self.HEIGHT = height
        self.PLATES_COUNT = plates_count
        self.NOISE_SCALE = noise_scale
        self.NOISE_STRENGTH = noise_strength

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.plates = [TectonicPlate(i) for i in range(self.PLATES_COUNT)]
        self.plate_map = -np.ones((self.HEIGHT, self.WIDTH), dtype=int)

    def generate(self):
        cost_map = np.full((self.HEIGHT, self.WIDTH), np.inf)
        pq = []

        # Seed каждой плиты
        for plate in self.plates:
            x = random.randint(0, self.WIDTH - 1)
            y = random.randint(0, self.HEIGHT - 1)
            self.plate_map[y, x] = plate.id
            cost_map[y, x] = 0.0
            heapq.heappush(pq, (0.0, x, y, plate.id))

        # Multi-source Dijkstra для роста плит
        while pq:
            cost, x, y, plate_id = heapq.heappop(pq)
            if cost > cost_map[y, x]:
                continue

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                    if self.plate_map[ny, nx] == -1:
                        noise_val = pnoise2(nx * self.NOISE_SCALE, ny * self.NOISE_SCALE, octaves=4)
                        step_cost = 1.0 + abs(noise_val) * self.NOISE_STRENGTH
                        new_cost = cost + step_cost
                        if new_cost < cost_map[ny, nx]:
                            cost_map[ny, nx] = new_cost
                            self.plate_map[ny, nx] = plate_id
                            heapq.heappush(pq, (new_cost, nx, ny, plate_id))

        return self.plate_map, self.plates

    def visualize(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(self.plate_map, cmap="tab20")
        plt.title(f"Тектонические плиты ({self.PLATES_COUNT})")
        plt.axis("off")
        plt.show()
