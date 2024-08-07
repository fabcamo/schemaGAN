import pickle

# Create a dictionary as an example

data = {
    'name': 'John Doe',
    'age': 30,
    'city': 'New York',
    'is_student': False,
    'hobbies': ['reading', 'writing', 'coding'],
    'grades': { 'math': 90, 'english': 85, 'science': 92 }
}

# Save the dictionary to a pickle file
# Define a path for the pickle file
save_path = 'D:\\schemaGAN\\tests'
with open(save_path + '/data.pkl', 'wb') as file:
    pickle.dump(data, file)

# Load the dictionary from the pickle file
with open(save_path + '/data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Print the loaded dictionary
print(loaded_data)