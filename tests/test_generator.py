# FIRST is the original schemaGAN GENERATOR ############################################################################
from schemaGAN.functions.arch_generator import define_generator

schemaGAN_generator = define_generator(image_shape=(32, 512, 1))
print('Original schemaGAN generator:............................................')
print(schemaGAN_generator.summary())

# NEXT IS THE MODULAR GENERATOR using schemaGAN parameters #############################################################
from models.generator import Generator_modular

generator = Generator_modular(input_size=(512, 32), no_inputs=1)
print('Modular generator:.........................................................')
print(generator.summary())

"""
# Last is Eleni's generator for the emulators ##########################################################################
from utils.gen_dis_eleni import Generator

OUTPUT_CHANNELS = 1
generator = Generator(input_shape=(256, 256, 4))
print('Elenis generator:..........................................................')
print(generator.summary())
"""
