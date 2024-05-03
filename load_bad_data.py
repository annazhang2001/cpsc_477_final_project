import tensorflow_datasets as tfds

# Load the dataset
dataset, info = tfds.load('bot_adversarial_dialogue', with_info=True)

# Get the train dataset
train_dataset = dataset['train']