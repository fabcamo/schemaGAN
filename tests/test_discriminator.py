from schemaGAN.functions.arch_discriminator import define_discriminator

schemaGAN_generator = define_discriminator(image_shape=(32, 512, 1))
#print(schemaGAN_generator.summary())


from models.generator import Discriminator

discriminator = Discriminator(input_size=(256, 256), no_inputs=3)
print(discriminator.summary())



from utils.generator_eleni import Discriminator, DiscriminatorTF
discriminator = Discriminator()
print(discriminator.summary())

discriminatorTF = DiscriminatorTF()
print(discriminatorTF.summary())

