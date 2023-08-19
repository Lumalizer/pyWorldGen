from world.WorldGenerator import WorldGenerator

wgen = WorldGenerator()
texture, heightMap, humidity = wgen.generate_procedural_world()
texture.save('texture.png')
heightMap.save('heightmap.png')
humidity.save('moisture.png')
