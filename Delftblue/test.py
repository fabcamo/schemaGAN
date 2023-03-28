import datetime
import tensorflow as tf
import pkg_resources
import sys

# Get current date and time
now = datetime.datetime.now()

# Get Python version and installed package versions
python_version = f"Python {sys.version}"
library_versions = '\n'.join([f"{d.project_name} {d.version}" for d in pkg_resources.working_set])

# Check number of GPUs available
num_gpus = len(tf.config.list_physical_devices('GPU'))

# Measure time to run
start_time = datetime.datetime.now()

# Add your code here

end_time = datetime.datetime.now()
duration = end_time - start_time

# Write to file
with open('results.txt', 'w') as file:
    file.write(f"Date and time: {now}\n")
    file.write(f"{python_version}\n")
    file.write(f"Installed libraries:\n{library_versions}\n")
    file.write(f"Number of GPUs in use: {num_gpus}\n")
    file.write(f"Time required to run: {duration}\n")
