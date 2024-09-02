import math
import os
import random
import sys
import json

from matplotlib.widgets import Button
from Utils.file_utils import (
    open_file,
    parse_file,
    parse_file_data_by_house,
    transform_file_data,
)
from Utils.my_math import my_len, my_max, my_min, my_sum
import matplotlib.pyplot as plt


class TrainingVisualization:
    def __init__(self, input_data, output_data, keys, features_info, all_thetas):
        self.input_data = input_data
        self.output_data = output_data
        self.all_thetas = all_thetas
        self.keys = keys
        self.index = 0
        self.features_info = features_info
        self.fig, self.ax = plt.subplots(2, 2)
        self.update_plot()
        self.create_buttons()

    def update_plot(self):
        key_selected = self.keys[self.index]

        for house, ax, color in [
            ("Slytherin", self.ax[0][0], "green"),
            ("Ravenclaw", self.ax[0][1], "blue"),
            ("Gryffindor", self.ax[1][0], "red"),
            ("Hufflepuff", self.ax[1][1], "yellow"),
        ]:
            ax.cla()
            course_data = [item[key_selected] for item in self.input_data]
            min_value = min(course_data)
            max_value = max(course_data)

            course_data.sort()

            ax.scatter(
                course_data,
                [
                    1 if self.output_data[i] == house else 0
                    for i in range(len(self.output_data))
                ],
                label=house,
                color=color,
            )

            probabilities = []
            for value in self.input_data:
                obj = {
                    feature: (
                        (value[feature] - self.features_info[feature]["mean"])
                        / self.features_info[feature]["std"]
                    )
                    for feature in self.keys
                }
                probabilities.append(sigmoid(dot_product(obj, all_theta[house])))

            ax.plot(course_data, probabilities)

            ax.set_xlim(
                min_value - abs(max_value) * 0.1,
                max_value + abs(max_value) * 0.1,
            )

        self.fig.canvas.draw_idle()

    def next_key(self, event):
        self.index = (self.index + 1) % len(self.keys)
        self.update_plot()

    def prev_key(self, event):
        self.index = (self.index - 1) % len(self.keys)
        self.update_plot()

    def create_buttons(self):
        axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.01, 0.1, 0.075])

        self.bnext = Button(axnext, "Next")
        self.bnext.on_clicked(self.next_key)

        self.bprev = Button(axprev, "Prev")
        self.bprev.on_clicked(self.prev_key)

    def next_key(self, event):
        self.index = (self.index + 1) % len(self.keys)
        self.update_plot()

    def prev_key(self, event):
        self.index = (self.index - 1) % len(self.keys)
        self.update_plot()

    def create_buttons(self):
        axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.01, 0.1, 0.075])

        self.bnext = Button(axnext, "Next")
        self.bnext.on_clicked(self.next_key)

        self.bprev = Button(axprev, "Prev")
        self.bprev.on_clicked(self.prev_key)


LEARNING_RATE = 0.01
ITERATION = 700

HOUSES = ["Slytherin", "Ravenclaw", "Gryffindor", "Hufflepuff"]


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def dot_product(output_element, theta):
    return sum(output_element[feature] * theta[feature] for feature in output_element)


if my_len(sys.argv) < 2:
    print("Need arg.")
else:
    if not os.path.isfile(sys.argv[1]):
        print("File not found.")
    else:
        DATASET_PATH = sys.argv[1]

        with open_file(sys.argv[1]) as file:
            print("Config Data Creating...")
            file_data = transform_file_data(parse_file(file))

            FEATURES = [
                # "Arithmancy",
                "Astronomy",
                "Herbology",
                "Defense Against the Dark Arts",
                "Divination",
                # "Muggle Studies",
                # "Ancient Runes",
                "History of Magic",
                # "Transfiguration",
                # "Potions",
                # "Care of Magical Creatures",
                # "Charms",
                "Flying",
            ]
            temp_data = []

            ALL_OUTPUT_DATA = []
            OUTPUT_DATA = []
            DATASET_LENGTH = len(file_data)

            INPUT_DATA = []
            INPUT_DATA_NO_NORMALIZE = []

            FEATURE_INFO = {
                feature: {"mean": 0, "std": 0, "weight": 0} for feature in FEATURES
            }

            for index in range(DATASET_LENGTH):
                INPUT_DATA.append({feature: 0 for feature in FEATURES})
                INPUT_DATA_NO_NORMALIZE.append({feature: 0 for feature in FEATURES})
                # INPUT_DATA[index]["biais"] = 1
                OUTPUT_DATA.append(file_data[index]["Hogwarts House"])
                for feature in FEATURES:
                    values = [
                        item[feature] if item[feature] else 0 for item in file_data
                    ]

                    min_val = min(values)
                    max_val = max(values)

                    mean = sum(values) / DATASET_LENGTH

                    sum_squares = 0
                    for i in range(len(values)):
                        sum_squares += (values[i] - mean) ** 2
                    std = sum_squares / (DATASET_LENGTH - 1)
                    std = std**0.5

                    FEATURE_INFO[feature]["std"] = std
                    FEATURE_INFO[feature]["mean"] = mean

                    INPUT_DATA_NO_NORMALIZE[index][feature] = values[index]
                    INPUT_DATA[index][feature] = (values[index] - mean) / std

                    # INPUT_DATA[index][feature] = (values[index] - min_val) / (max_val - min_val)

            FEATURES_NUMBER = my_len(FEATURES)
            NUMBER_ITEM = my_len(file_data)

            ALL_OUTPUT_DATA = OUTPUT_DATA.copy()

            print("Config Data Created.")
            print("Split Data For testing...")

            INPUT_TESTING_DATA = []
            OUTPUT_TESTING_DATA = []

            indices = list(range(round(DATASET_LENGTH * 0.2)))

            while len(indices):
                index = indices.pop()
                INPUT_TESTING_DATA.append(INPUT_DATA.pop(index))
                OUTPUT_TESTING_DATA.append(OUTPUT_DATA.pop(index))

            DATASET_LENGTH = len(INPUT_DATA)

            all_theta = {
                house: {feature: 0 for feature in FEATURES} for house in HOUSES
            }
            print("Start Training.")
            for house in HOUSES:
                print(f"Training {house}...")
                theta = {feature: 0 for feature in FEATURES}
                OUTPUT_DATA_ENCODED = [
                    1 if value == house else 0 for value in OUTPUT_DATA
                ]
                for _ in range(ITERATION):
                    all_errors = []
                    gradients = {feature: 0 for feature in theta}
                    for i in range(DATASET_LENGTH):
                        h = sigmoid(dot_product(INPUT_DATA[i], theta))
                        all_errors.append(
                            1
                            if (h > 0.5 and OUTPUT_DATA_ENCODED[i] == 1)
                            or (h <= 0.5 and OUTPUT_DATA_ENCODED[i] == 0)
                            else 0
                        )
                        for feature in theta:
                            gradients[feature] += (
                                h - OUTPUT_DATA_ENCODED[i]
                            ) * INPUT_DATA[i][feature]
                    for feature in theta:
                        theta[feature] -= (
                            LEARNING_RATE * (1 / DATASET_LENGTH) * gradients[feature]
                        )
                all_theta[house] = theta

                print(f"Training {house} Finished.")

            probabilities = []
            for i in range(len(INPUT_TESTING_DATA)):
                house_probs = {
                    house: sigmoid(dot_product(INPUT_TESTING_DATA[i], all_theta[house]))
                    for house in HOUSES
                }
                probabilities.append(max(house_probs, key=house_probs.get))

            precision = (
                len(
                    [
                        1
                        for i in range(len(OUTPUT_TESTING_DATA))
                        if probabilities[i] == OUTPUT_TESTING_DATA[i]
                    ]
                )
                * 100
                / len(OUTPUT_TESTING_DATA)
            )

            try:
                merged_data = all_theta.copy()
                merged_data.update(FEATURE_INFO)
                
                with open("theta_values.json", "w") as fichier:
                    json.dump(merged_data, fichier, indent=4)
            except IOError as e:
                print(f"Erreur d'écriture dans le fichier: {e}")
            

            # print(
            #     len(
            #         [
            #             1
            #             for i in range(len(OUTPUT_TESTING_DATA))
            #             if probabilities[i] == OUTPUT_TESTING_DATA[i]
            #         ]
            #     )
            # )
            # print(len(OUTPUT_TESTING_DATA))

            TrainingVisualization(
                INPUT_DATA_NO_NORMALIZE,
                ALL_OUTPUT_DATA,
                FEATURES,
                FEATURE_INFO,
                all_theta,
            )
            # print(FEATURE_INFO)

            print(precision)

            # plt.show()


# return all_theta
