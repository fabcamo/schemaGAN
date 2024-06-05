from schemaGAN.functions.arch_generator import define_generator

schemaGAN_generator = define_generator(image_shape=(32, 512, 1))
print(schemaGAN_generator.summary())

from models.generator import Generator_modular

generator = Generator_modular(input_shape=(32, 512, 4))
print(generator.summary())

from utils.gen_dis_eleni import Generator
OUTPUT_CHANNELS = 1
generator = Generator(input_shape=(256, 256, 4))
print(generator.summary())



from models.generator import Generator_modular as GeneratorA
from utils.gen_dis_eleni import Generator as GeneratorB
import io

# Initialize the models
generator_a = GeneratorA(input_shape=(256, 256, 1))
generator_b = GeneratorB(input_shape=(256, 256, 1))

# Capture the summaries
def capture_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str


summary_a = capture_summary(generator_a)
summary_b = capture_summary(generator_b)


# Function to compare summaries line by line
def compare_summaries(summary1, summary2):
    summary1_lines = summary1.splitlines()
    summary2_lines = summary2.splitlines()

    # Ensure the summaries are the same length for comparison
    max_length = max(len(summary1_lines), len(summary2_lines))
    summary1_lines.extend([''] * (max_length - len(summary1_lines)))
    summary2_lines.extend([''] * (max_length - len(summary2_lines)))

    differences = []
    for i, (line1, line2) in enumerate(zip(summary1_lines, summary2_lines)):
        if line1 != line2:
            differences.append((i, line1, line2))

    return differences


differences = compare_summaries(summary_a, summary_b)

# Print the differences
if differences:
    print("\nDifferences found between the two model summaries:\n")
    for index, line_a, line_b in differences:
        print(f"Line {index + 1}:")
        print(f"Model Modular: {line_a}")
        print(f"Model Eleni: {line_b}")
        print()
else:
    print("The summaries of the two models are identical.")
