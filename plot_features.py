import json
from matplotlib import pyplot as plt


def plot():
    with open('data_with_colors.json', 'r') as f:
        data = json.load(f)
        index = 0
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        zdata = []
        ydata = []
        xdata = []
        colors = []
        for i in data.keys():
            index = index + 1
            if index == 1000:
                break
            zdata.append(data[i]["horizontal_histogram_projection"])
            ydata.append(data[i]["vertical_histogram_projection"])
            xdata.append(data[i]["aspect_ratio"])
            colors.append(data[i]['color'])
        ax.scatter3D(xdata, ydata, zdata, c=colors)
        plt.show()