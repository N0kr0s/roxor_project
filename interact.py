import pygame
import sys
import h5py
import os
import numpy as np
import noise

# --------------------
# Настройки ID и цветов
# --------------------
GROUND_ID = {
    1: "grass",
    2: "sand",
    3: "water",
    4: "mountain",
    5: "tundra"
}

OBJECT_ID = {
    0: "empty",
    1: "tree",
    2: "bush",
    3: "rock",
    4: "animal"
}

GROUND_COLORS = {
    1: (34,139,34),   # grass
    2: (237,201,175), # sand
    3: (0,0,255),     # water
    4: (120,110,100), # mountain
    5: (198,226,255)  # tundra
}

OBJECT_COLORS = {
    0: (0,0,0),       # empty
    1: (0,100,0),     # tree
    2: (0,200,0),     # bush
    3: (128,128,128), # rock
    4: (255,0,0)      # animal
}

CHUNK_SIZE = 32
chunk_cache = {}

# --------------------
# Генерация чанка с ID
# --------------------
def generate_heightmap(seed_x, seed_y):
    heightmap = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
    scale = 16.0
    for y in range(CHUNK_SIZE):
        for x in range(CHUNK_SIZE):
            nx = (seed_x*CHUNK_SIZE+x)/scale
            ny = (seed_y*CHUNK_SIZE+y)/scale
            h = noise.pnoise2(nx, ny, octaves=4, persistence=0.5, lacunarity=2.0)
            heightmap[y,x] = h
    return heightmap

def generate_chunk(world_x, world_y, biome_name):
    key = (world_x, world_y)
    if key in chunk_cache:
        return chunk_cache[key]

    # Ground слой: ID тайлов
    ground_layer = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
    for y in range(CHUNK_SIZE):
        for x in range(CHUNK_SIZE):
            if biome_name == "desert":
                ground_layer[y,x] = 2
            elif biome_name == "tundra":
                ground_layer[y,x] = 5
            elif biome_name == "mountain":
                ground_layer[y,x] = 4
            else:
                ground_layer[y,x] = 1

    # Objects слой: изначально пустой
    objects_layer = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)

    chunk_cache[key] = {
        "ground": ground_layer,
        "objects": objects_layer
    }
    return chunk_cache[key]

# --------------------
# Игровой класс
# --------------------
class Game:
    def __init__(self):
        self.MAP_SIZE = 64
        self.WINDOW_SIZE = (1200,800)
        self.PNG_PATH = "world_map_gradient.png"
        self.H5_PATH = "world_data.h5"

        if not os.path.exists(self.PNG_PATH) or not os.path.exists(self.H5_PATH):
            print("❌ Нет PNG/H5")
            sys.exit()

        pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Chunk + World Map with IDs")
        self.clock = pygame.time.Clock()

        # Глобальная карта
        self.image_original = pygame.image.load(self.PNG_PATH).convert()
        self.image_original = pygame.transform.scale(self.image_original, (self.MAP_SIZE, self.MAP_SIZE))

        # HDF5
        self.h5 = h5py.File(self.H5_PATH, "r")
        self.biome = self.h5["biome"]
        self.height = self.h5["height"]

        # Текущий чанк
        self.chunk_pos = (0,0)
        self.chunk_data = None
        self.load_chunk(self.chunk_pos)

        # Игрок
        self.player_x = CHUNK_SIZE//2
        self.player_y = CHUNK_SIZE//2

        self.run()

    # --------------------
    def load_chunk(self, pos):
        x, y = pos
        raw_biome = self.biome[y%self.MAP_SIZE, x%self.MAP_SIZE]
        biome_name = raw_biome.decode("utf-8") if isinstance(raw_biome, bytes) else str(raw_biome)
        self.chunk_data = generate_chunk(x, y, biome_name)

    # --------------------
    def move_player(self, dx, dy):
        nx = self.player_x + dx
        ny = self.player_y + dy
        if 0 <= nx < CHUNK_SIZE and 0 <= ny < CHUNK_SIZE:
            self.player_x, self.player_y = nx, ny
        else:
            cx, cy = self.chunk_pos
            if nx < 0: cx -=1; nx = CHUNK_SIZE-1
            if nx >= CHUNK_SIZE: cx +=1; nx = 0
            if ny < 0: cy -=1; ny = CHUNK_SIZE-1
            if ny >= CHUNK_SIZE: cy +=1; ny = 0
            self.chunk_pos = (cx, cy)
            self.load_chunk(self.chunk_pos)
            self.player_x, self.player_y = nx, ny

    # --------------------
    def place_tree(self):
        self.chunk_data["objects"][self.player_y,self.player_x] = 1  # tree

    def cut_tree(self):
        self.chunk_data["objects"][self.player_y,self.player_x] = 0  # empty

    def inspect_cell(self):
        gx, gy = self.chunk_pos
        g_id = self.chunk_data["ground"][self.player_y, self.player_x]
        o_id = self.chunk_data["objects"][self.player_y, self.player_x]
        print(f"Чанк {gx},{gy} | Игрок: {self.player_x},{self.player_y}")
        print(f"Ground ID: {g_id} ({GROUND_ID[g_id]}) | Object ID: {o_id} ({OBJECT_ID[o_id]})")

    # --------------------
    def draw_chunk(self, rect):
        surf = pygame.Surface((CHUNK_SIZE, CHUNK_SIZE))
        ground = self.chunk_data["ground"]
        objects = self.chunk_data["objects"]

        for y in range(CHUNK_SIZE):
            for x in range(CHUNK_SIZE):
                color = GROUND_COLORS[ground[y,x]]
                obj_id = objects[y,x]
                if obj_id !=0:
                    color = OBJECT_COLORS[obj_id]
                surf.set_at((x,y), color)

        surf.set_at((self.player_x,self.player_y),(255,0,0))  # игрок

        surf = pygame.transform.scale(surf,(rect.width, rect.height))
        self.screen.blit(surf, rect.topleft)

    # --------------------
    def draw_world_map(self, rect, mouse_pos):
        scale = min(rect.width, rect.height)/self.MAP_SIZE
        disp_size = int(self.MAP_SIZE*scale)
        offset_x = rect.x + (rect.width - disp_size)//2
        offset_y = rect.y + (rect.height - disp_size)//2

        image_scaled = pygame.transform.scale(self.image_original, (disp_size, disp_size))
        self.screen.blit(image_scaled,(offset_x, offset_y))

        # подсветка под мышью
        mx, my = mouse_pos
        map_x = int((mx - offset_x)/scale)
        map_y = int((my - offset_y)/scale)
        if 0<=map_x<self.MAP_SIZE and 0<=map_y<self.MAP_SIZE:
            size_mouse = max(1,int(scale))
            highlight_mouse = pygame.Surface((size_mouse,size_mouse))
            highlight_mouse.fill((255,0,255))
            self.screen.blit(highlight_mouse,(offset_x+int(map_x*scale), offset_y+int(map_y*scale)))

        # подсветка текущего чанка игрока
        cx, cy = self.chunk_pos
        size_chunk = max(1,int(scale))
        highlight_chunk = pygame.Surface((size_chunk,size_chunk))
        highlight_chunk.fill((200,0,200))
        self.screen.blit(highlight_chunk,(offset_x+int(cx*scale), offset_y+int(cy*scale)))

        return map_x, map_y

    # --------------------
    def run(self):
        running = True
        while running:
            win_w, win_h = self.screen.get_size()
            left_rect = pygame.Rect(0,0,win_w//2,win_h)
            right_rect = pygame.Rect(win_w//2,0,win_w//2,win_h)
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running=False
                elif event.type==pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w,event.h),pygame.RESIZABLE)
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_w: self.move_player(0,-1)
                    elif event.key==pygame.K_s: self.move_player(0,1)
                    elif event.key==pygame.K_a: self.move_player(-1,0)
                    elif event.key==pygame.K_d: self.move_player(1,0)
                    elif event.key==pygame.K_f: self.place_tree()
                    elif event.key==pygame.K_x: self.cut_tree()
                    elif event.key==pygame.K_z: self.inspect_cell()
                elif event.type==pygame.MOUSEBUTTONDOWN:
                    map_x, map_y = self.draw_world_map(right_rect, mouse_pos)
                    if 0<=map_x<self.MAP_SIZE and 0<=map_y<self.MAP_SIZE:
                        self.chunk_pos = (map_x, map_y)
                        self.player_x = CHUNK_SIZE//2
                        self.player_y = CHUNK_SIZE//2
                        self.load_chunk(self.chunk_pos)

            self.screen.fill((0,0,0))
            self.draw_chunk(left_rect)
            self.draw_world_map(right_rect, mouse_pos)

            pygame.display.flip()
            self.clock.tick(60)

        self.h5.close()
        pygame.quit()
        sys.exit()