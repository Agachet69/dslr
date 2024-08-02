import sys
import os
from Utils.file_utils import (
    open_file,
    parse_file,
    # transform_file_data,
)
from Utils.my_math import my_len
import json
import math
import pandas

HOUSES = ["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]

FEATURES =  [
                "Astronomy",
                "Herbology",
                "Defense Against the Dark Arts",
                "Divination",
                "History of Magic",
                "Flying",
            ]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def dot_product(output_element, theta):
    return sum(output_element[feature] * theta[feature] for feature in output_element if output_element[feature])

def transform_file_data(file_data):
    '''
        Transform a file data like : {First Name: [value1, value2, value3, ...], Astronomy: [value1, value2, value3, ...]}
        In a file data like : [{First Name: value, Astronomy: value}, {First Name: value, Astronomy: value}]
    '''
    new_file_data = []


    for index in range(my_len(file_data['Index'])):
        obj = {key: file_data[key][index] for key in file_data if key in FEATURES}
        new_file_data.append(obj)
    
    return new_file_data

if len(sys.argv) < 2:
    print("Need dataset argument.")
else :
    try:
        with open("theta_values.json", "r") as fichier:
            theta_data = json.load(fichier)
        if (theta_data['Slytherin'] and theta_data['Ravenclaw'] and theta_data['Gryffindor'] and theta_data['Hufflepuff']):
            print('Theta receive.')
    except FileNotFoundError as e:
        exit(f"The file was not found. Please run the train before run the predict.")
    except:
        exit(f"The json file has the wrong format.")

    if not os.path.isfile(sys.argv[1]):
        print("File not found.")
    else:
        # try:
            DATASET_PATH = sys.argv[1]

            with open_file(sys.argv[1]) as file:
                file_data = transform_file_data(parse_file(file))

            for house in HOUSES:
                probabilities = []
                for i in range(len(file_data)):
                    house_probs = {
                        house: sigmoid(dot_product(file_data[i], theta_data[house]))
                        for house in HOUSES
                    }
                    probabilities.append(max(house_probs, key=house_probs.get))

            df = pandas.DataFrame(probabilities, columns=['Hogwarts House'])
            df.to_csv('houses.csv', index_label='Index')
            print('You can find your prediction in the houses.csv file.')
        
        # except:
        #     exit('Error with tests datas.')
