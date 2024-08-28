# Original schemaGAN discriminator #####################################################################################
from schemaGAN.functions.arch_discriminator import define_discriminator

schemaGAN_generator = define_discriminator(image_shape=(32, 512, 1))
print('Original schemaGAN discriminator')
print(schemaGAN_generator.summary())

# Modular discriminator ################################################################################################
from models.discriminator import Discriminator_modular

discriminator = Discriminator_modular(input_size=(32, 512), no_inputs=1)
print('Modular discriminator')
print(discriminator.summary())

# Eleni's discriminator for the emulators ##############################################################################

#from utils.gen_dis_eleni import Discriminator, DiscriminatorTF
#discriminator = Discriminator()
#print(discriminator.summary())

#discriminatorTF = DiscriminatorTF()
#print(discriminatorTF.summary())

