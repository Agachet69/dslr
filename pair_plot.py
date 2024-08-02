import math
import os
import sys

from matplotlib.widgets import Button
from Utils.file_utils import (
    open_file,
    parse_file,
    parse_file_data_by_house,
    transform_file_data,
)
from Utils.my_math import my_len, my_max, my_min
import matplotlib.pyplot as plt

if my_len(sys.argv) < 2:
    print("Need arg.")
else:
    if not os.path.isfile(sys.argv[1]):
        print("File not found.")
    else:
        DATASET_PATH = sys.argv[1]

        with open_file(sys.argv[1]) as file:
            file_data = transform_file_data(parse_file(file))
            
            courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
            
            number_of_courses = len(courses)
            
            fig, ax = plt.subplots(number_of_courses, number_of_courses)
            print(file_data)
            
            for key_x_index, key_x in enumerate(courses):
                is_after_same_key = False
                for key_y_index, key_y in enumerate(courses):                    
                    for house, color in [
                        ("Slytherin", "green"),
                        ("Ravenclaw", "blue"),
                        ("Gryffindor", "red"),
                        ("Hufflepuff", "yellow"),
                    ]:
                        
                        if is_after_same_key:
                            break
                        
                        x_data = [(item[key_x] if item[key_x] else None) for item in file_data if item["Hogwarts House"] == house]
                        y_data = [(item[key_y] if item[key_y] else None) for item in file_data if item["Hogwarts House"] == house]
                        
                        none_index = [index for index, value in enumerate(x_data) if value == None] + [index for index, value in enumerate(y_data) if value == None]
                        
                        none_index.sort()
                        
                        while len(none_index):
                            index = none_index.pop(len(none_index) - 1)
                            x_data.pop(index)
                            y_data.pop(index)
                        
                        
                        if key_x == key_y:
                            item_number = len(x_data)
                            class_number = round(1 + math.log2(item_number))

                            min_value = my_min(x_data)
                            max_value = my_max(x_data)

                            ax[key_x_index][key_y_index].hist([value for value in x_data if value], bins=class_number, range=[min_value, max_value], color=color, alpha=0.5, label=house)
                        else:
                            ax[key_x_index][key_y_index].scatter(x_data, y_data, color=color, alpha=.5)
                        
                    ax[key_x_index][key_y_index].set_xticks([])
                    ax[key_x_index][key_y_index].set_yticks([])
                    
                    if key_x_index == number_of_courses - 1:
                        ax[key_x_index][key_y_index].set_xlabel(key_y.replace(' ', '\n'), fontsize=5)
                    if key_y_index == 0:
                        ax[key_x_index][key_y_index].set_ylabel(key_x.replace(' ', '\n'), fontsize=5)
                    
                    if not is_after_same_key:
                        is_after_same_key = key_x == key_y
                    
                    
                    # ax[key_x_index][key_y_index].set_axis_off()
                    
            plt.subplots_adjust(bottom=.1, left=.05, top=.99, right=.99)
            plt.show()
            