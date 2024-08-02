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

class HistogramCourses:
    def __init__(self, houses_data, keys):
        self.houses_data = houses_data
        self.keys = keys
        self.index = 0
        self.fig, self.ax  = plt.subplots()
        self.update_plot()
        self.create_buttons()

    def update_plot(self):
        key_selected = self.keys[self.index]
        
        self.ax.cla()
        
        for house, data, color in [
            ("Slytherin", self.houses_data["Slytherin"], "green"),
            ("Ravenclaw", self.houses_data["Ravenclaw"], "blue"),
            ("Gryffindor", self.houses_data["Gryffindor"], "red"),
            ("Hufflepuff", self.houses_data["Hufflepuff"], "yellow"),
        ]:
            course_data = [item[key_selected] for item in data]
            
            item_number = len(course_data)
            class_number = round(1 + math.log2(item_number))
            
            min_value = my_min(course_data)
            max_value = my_max(course_data)
            
            self.ax.hist([value for value in course_data if value], bins=class_number, range=[min_value, max_value], color=color, alpha=0.5, label=house)
                
        self.ax.legend()
        self.ax.set_title(f"Histogram for {key_selected}")
        self.ax.set_xlim(
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


if my_len(sys.argv) < 2:
    print("Need arg.")
else:
    if not os.path.isfile(sys.argv[1]):
        print("File not found.")
    else:
        DATASET_PATH = sys.argv[1]

        with open_file(sys.argv[1]) as file:
            file_data = parse_file(file)

            houses_data = parse_file_data_by_house(file_data)
            file_data = transform_file_data(file_data)

            # gryffindor_data = get_houses_std_value(houses_data["Gryffindor"])
            # ravenclaw_data = get_houses_std_value(houses_data["Ravenclaw"])
            # slytherin_data = get_houses_std_value(houses_data["Slytherin"])
            # hufflepuff_data = get_houses_std_value(houses_data["Hufflepuff"])

            # print(houses_data)
            # for house in houses_data:
                # data = houses_data[house]
                # print(data)
                # class_number = 1 + math.log2()
            # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # fig.suptitle("Courses grade for each Houses")

            # ax1.plot([key.split(' ')[0] for key in gryffindor_data], [gryffindor_data[key] for key in gryffindor_data], color='red' )
            # ax1.tick_params(axis='x', labelrotation=90)
            # ax1.set_title("Gryffindor Std Courses Grade")

            # ax2.plot([key.split(' ')[0] for key in ravenclaw_data], [ravenclaw_data[key] for key in ravenclaw_data], color='blue' )
            # ax2.tick_params(axis='x', labelrotation=90)
            # ax2.set_title("Ravenclaw Std Courses Grade")

            # ax3.plot([key.split(' ')[0] for key in slytherin_data], [slytherin_data[key] for key in slytherin_data], color='green' )
            # ax3.tick_params(axis='x', labelrotation=90)
            # ax3.set_title("Slytherin Std Courses Grade")

            # ax4.plot([key.split(' ')[0] for key in hufflepuff_data], [hufflepuff_data[key] for key in hufflepuff_data], color='yellow' )
            # ax4.tick_params(axis='x', labelrotation=90)
            # ax4.set_title("Hufflepuff Std Courses Grade")
            keys_possible = [
                key
                for key in file_data[0]
                if isinstance(file_data[0][key], float) and key != "Index"
            ]
            scatter_plot = HistogramCourses(houses_data, keys_possible)
            plt.show()
