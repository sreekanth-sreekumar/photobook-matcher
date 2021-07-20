from flask import Flask
from numpy import random
from dialogue_manager.system import getListenerPredictions, getSpeakerMessage

app = Flask(__name__)

image_set = []
top_scores, top_images = []
curr_image_index = 0
top_counter = 0

@app.route('/get_images')
def get_images():
    global curr_image_index, top_scores, top_images, image_set
    image_sets = []
    with open('./image_file.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            image_set = line.split(' ')
            image_sets.append(image_set)
        n = random.randint(len(image_sets))
        image_set = ' '.join(image_sets[n])
        curr_image_index = 0
        top_scores, top_images = []
        return image_set

@app.route('/get_answer/<message>', methods=['GET'])
def get_answer(message):
    global curr_image_index, top_scores, top_images, image_set
    if message == "Yes":
        message = "You have chosen Image number " + str(top_images[curr_image_index] + 1)
    elif message == "No":
        curr_image_index += 1
        message = getSpeakerMessage(image_set, message, top_images[curr_image_index])
    else:
        top_scores, top_images = getListenerPredictions(image_set, message)
        message = "Did you mean " + getSpeakerMessage(image_set, message, top_images[curr_image_index])
        
    