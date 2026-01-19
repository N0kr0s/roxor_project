from tectonics import TectonicsGenerator
from world import WorldGenerator
from interact import Game

def generate():
    WIDTH = 64
    HEIGHT = 64
    PLATES_COUNT = 12

    # ------------------------------
    # Генерация тектоники
    # ------------------------------
    tectonics_gen = TectonicsGenerator(width=WIDTH, height=HEIGHT, plates_count=PLATES_COUNT)
    plate_map, plates = tectonics_gen.generate()
    tectonics_gen.visualize()

    # ------------------------------
    # Генерация мира
    # ------------------------------
    world_gen = WorldGenerator(width=WIDTH, height=HEIGHT, plate_map=plate_map, plates=plates)
    world_gen.create_base_world()
    world_gen.detect_boundaries()
    world_gen.apply_terrain()
    world_gen.assign_biomes()

    # ------------------------------
    # Экспорт
    # ------------------------------
    world_gen.export_hdf5("world_data.h5")
    world_gen.export_png("world_map_gradient.png")
    # world_gen.export_heightmap_png("heightmap.png")

if __name__ == "__main__":
    if input("Generate world? (y/n)").lower() == "y":
        generate()
        Game()
    else:
        Game()