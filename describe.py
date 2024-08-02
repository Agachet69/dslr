import math
import sys
import os

from Utils.file_utils import open_file, parse_file
from Utils.my_math import my_floor, my_len, my_max, my_min, my_round, my_sum

def calculate_table_values(file_data):
    table_values = {}
    for key in file_data:
        if isinstance(file_data[key][0], float) and key != 'Index':
            # print(key)
            table_values[key] = {}
            
            count = my_len(file_data[key])
            mean = my_sum(file_data[key]) / count
            
            _variance = my_sum([(value-mean)**2 for value in file_data[key] if value]) / count
            
            std = math.sqrt(_variance / (count-1))
            
            min_value = my_min(file_data[key])
            max_value = my_max(file_data[key])
        
            _sorted_tab = [value for value in file_data[key] if value]
            number_none = len([value for value in file_data[key] if value == None])
            _sorted_tab.sort()
            for _ in range(number_none):
                _sorted_tab.insert(0, None)
            
            if not my_len(_sorted_tab) % 2:
                second_quartile = (_sorted_tab[my_floor(my_len(_sorted_tab) / 2)] + _sorted_tab[my_round(my_len(_sorted_tab) / 2 + .5)]) / 2
                _first_half = _sorted_tab[:my_floor(my_len(_sorted_tab) / 2)]
                _second_half = _sorted_tab[my_floor(my_len(_sorted_tab) / 2 + .5):]
            else:
                second_quartile = _sorted_tab[my_floor(my_len(_sorted_tab) / 2)]
                _first_half = _sorted_tab[:my_floor(my_len(_sorted_tab) / 2)]
                _second_half = _sorted_tab[my_floor(my_len(_sorted_tab) / 2 + .5):]
            if not my_len(_first_half) % 2:
                first_quartile = (_first_half[my_floor(my_len(_first_half) / 2)] + _first_half[my_round(my_len(_first_half) / 2 + .5)]) / 2
            else:
                first_quartile = _first_half[my_floor(my_len(_first_half))]
            if not my_len(_second_half) % 2:
                third_quartile = (_second_half[my_floor(my_len(_second_half) / 2)] + _second_half[my_round(my_len(_second_half) / 2 + .5)]) / 2
            else:
                third_quartile = _second_half[my_floor(my_len(_second_half))]
            
            table_values[key]["Count"] = count
            table_values[key]["Mean"] = mean
            table_values[key]["Std"] = std
            table_values[key]["Min"] = min_value
            table_values[key]["25%"] = first_quartile
            table_values[key]["50%"] = second_quartile
            table_values[key]["75%"] = third_quartile
            table_values[key]["Max"] = max_value
    return table_values

def display_summary_statistics(table_values):
    headers = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    
    obj = {key: [table_values[v][key] for v in table_values] for key in headers}
    
    print(11 * " " + " ".join(["{:15}".format(key.split(' ')[0]) for key in table_values]))
    
    for key in obj:
        print("{:10}".format(key), " ".join(["{:15}".format("{:15f}".format(v)) for v in obj[key]]))

if my_len(sys.argv) < 2:
    print("Need arg.")
else:
    if not os.path.isfile(sys.argv[1]):
        print("File not found.")
    else:
        DATASET_PATH = sys.argv[1]
        
        with open_file(sys.argv[1]) as file:
        
            file_data = parse_file(file)
            
            table_values = calculate_table_values(file_data)
            
            display_summary_statistics(table_values)
