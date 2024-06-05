from schemaGAN.functions.arch_discriminator import define_discriminator

schemaGAN_generator = define_discriminator(image_shape=(32, 512, 1))
print(schemaGAN_generator.summary())


from models.generator import Discriminator_modular

discriminator = Discriminator_modular(input_size=(32, 512), no_inputs=1)
print(discriminator.summary())



from utils.gen_dis_eleni import Discriminator, DiscriminatorTF
discriminator = Discriminator()
#print(discriminator.summary())

discriminatorTF = DiscriminatorTF()
#print(discriminatorTF.summary())

