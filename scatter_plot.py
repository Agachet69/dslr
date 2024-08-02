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


class ScatterPlotCourses:
    def __init__(self, houses_data, keys):
        self.houses_data = houses_data
        self.keys = keys
        self.indexKeyOne = 0
        self.indexKeyTwo = 1
        self.fig, self.ax = plt.subplots()
        self.scatterSlytherin = None
        self.scatterRavenclaw = None
        self.scatterGryffindor = None
        self.scatterHufflepuff = None
        self.update_plot()
        self.create_buttons()

    def update_plot(self):
        key1_selected = self.keys[self.indexKeyOne]
        key2_selected = self.keys[self.indexKeyTwo]

        if self.scatterSlytherin:
            self.scatterSlytherin.remove()
        if self.scatterRavenclaw:
            self.scatterRavenclaw.remove()
        if self.scatterGryffindor:
            self.scatterGryffindor.remove()
        if self.scatterHufflepuff:
            self.scatterHufflepuff.remove()

        key1_data_min_value = None
        key1_data_max_value = None
        
        key2_data_min_value = None
        key2_data_max_value = None

        for house, data, color in [
            ("Slytherin", self.houses_data["Slytherin"], "green"),
            ("Ravenclaw", self.houses_data["Ravenclaw"], "blue"),
            ("Gryffindor", self.houses_data["Gryffindor"], "red"),
            ("Hufflepuff", self.houses_data["Hufflepuff"], "yellow"),
        ]:
            key2_data = [item[key2_selected] for item in data]
            key1_data = [item[key1_selected] for item in data]

            if key1_data_min_value == None or my_min(key1_data) < key1_data_min_value:
                key1_data_min_value = my_min(key1_data)
            if key1_data_max_value == None or my_max(key1_data) > key1_data_max_value:
                key1_data_max_value = my_max(key1_data)
                
            if key2_data_min_value == None or my_min(key2_data) < key2_data_min_value:
                key2_data_min_value = my_min(key2_data)
            if key2_data_max_value == None or my_max(key2_data) > key2_data_max_value:
                key2_data_max_value = my_max(key2_data)

            if house == "Slytherin":
                self.scatterSlytherin = self.ax.scatter(
                    key2_data, key1_data, label=house, color=color
                )
            if house == "Ravenclaw":
                self.scatterRavenclaw = self.ax.scatter(
                    key2_data, key1_data, label=house, color=color
                )
            if house == "Gryffindor":
                self.scatterGryffindor = self.ax.scatter(
                    key2_data, key1_data, label=house, color=color
                )
            if house == "Hufflepuff":
                self.scatterHufflepuff = self.ax.scatter(
                    key2_data, key1_data, label=house, color=color
                )
        self.ax
        self.ax.legend()
        self.ax.set_title(f"Scatter plot for {key1_selected} by {key2_selected} of each houses")
        self.ax.set_xlim(
            key2_data_min_value - abs(key2_data_max_value)*.1,
            key2_data_max_value + abs(key2_data_max_value)*.1
        )
        self.ax.set_ylim(
            key1_data_min_value - abs(key1_data_max_value)*.1,
            key1_data_max_value + abs(key1_data_max_value)*.1
        )
        self.ax.set_xlabel(key2_selected)
        self.ax.set_ylabel(key1_selected)
        self.fig.canvas.draw_idle()

    def set_response_index(self, event):
        self.indexKeyOne = 3
        self.indexKeyTwo = 1
        self.update_plot()

    def next_key1(self, event):
        self.indexKeyOne = (self.indexKeyOne + 1) % len(self.keys)
        if self.indexKeyOne == self.indexKeyTwo:
            self.next_key1(event=None)
        self.update_plot()

    def prev_key1(self, event):
        self.indexKeyOne = (self.indexKeyOne - 1) % len(self.keys)
        if self.indexKeyOne == self.indexKeyTwo:
            self.prev_key1(event=None)
        self.update_plot()
        
    def next_key2(self, event):
        self.indexKeyTwo = (self.indexKeyTwo + 1) % len(self.keys)
        if self.indexKeyOne == self.indexKeyTwo:
            self.next_key2(event=None)
        self.update_plot()

    def prev_key2(self, event):
        self.indexKeyTwo = (self.indexKeyTwo - 1) % len(self.keys)
        if self.indexKeyOne == self.indexKeyTwo:
            self.prev_key2(event=None)
        self.update_plot()

    def create_buttons(self):
        axprev1 = plt.axes([0.1, 0.01, 0.1, 0.075])
        axnext1 = plt.axes([0.21, 0.01, 0.1, 0.075])
        
        axprev2 = plt.axes([0.7, 0.01, 0.1, 0.075])
        axnext2 = plt.axes([0.81, 0.01, 0.1, 0.075])
# 
        self.bprev1 = Button(axprev1, "Prev Key One")
        self.bprev1.on_clicked(self.prev_key1)
# 
        self.bnext1 = Button(axnext1, "Next Key One")
        self.bnext1.on_clicked(self.next_key1)
        # 
        self.bprev2 = Button(axprev2, "Prev Key Two")
        self.bprev2.on_clicked(self.prev_key2)
# 
        self.bnext2 = Button(axnext2, "Next Key Two")
        self.bnext2.on_clicked(self.next_key2)
        
        axres = plt.axes([0.81, 0.99 - .075, 0.1, 0.075])
        
        self.bres = Button(axres, "Set Response Index")
        self.bres.on_clicked(self.set_response_index)


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

            keys_possible = [
                key
                for key in file_data[0]
                if isinstance(file_data[0][key], float) and key != "Index"
            ]
            current_key_index = 0

            scatter_plot = ScatterPlotCourses(houses_data, keys_possible)
            # scatter_plot_house = ScatterPlotHouses(houses_data)

            plt.show()
