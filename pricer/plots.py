import random
from matplotlib import (
    pyplot as plt,
    colors as mcolors,
)


def adjust_results(results, error_reference):
    for result in results:
        for key, [value, cpu_time] in results[result].items():
            value = abs(value-error_reference)
            if value == 0:
                value += pow(10.0, -16)
            results[result][key] = [value, cpu_time]
    return results


def error_plots(results, grid):
    error_reference = list(results[0].values())[-1][0]
    results = adjust_results(results.copy(), error_reference)

    colours = mcolors.CSS4_COLORS
    colours_to_delete = ["white", "snow", "ghostwhite", "mintcream", "floralwhite", "whitesmoke"]
    try:
        random.shuffle([colours.pop(key) for key in colours_to_delete])
    except KeyError:
        pass
    colours = list(colours.values())
    colours = list(mcolors.TABLEAU_COLORS.values())+colours
    marker = ['o', '*', '.', 'x', '+', 's', 'D', 'p', 'h', '^', '<', '>', '2']
    linestyles = [':', '-.', '--', ':', '-.', '--', ':', '-.', '--', ':', '-.', '--', ':', '-.', '--']
    random.shuffle(marker)
    # fig1, (price_plot, cpu_plot) = plt.subplots(nrows=1, ncols=2) - for subplots
    for i in results:
        error_values = [x[0] for x in list(results[i].values())]
        plt.figure(0)
        plt.loglog(grid, error_values, linestyle=linestyles[i], color=colours[i], marker=marker[i],
                   label=list(results[i].keys())[i][0])
        # cpu_times = [x[1] for x in list(results[i].values())]
        # plt.figure(1)
        # plt.loglog(cpu_times, error_values, linestyle='--', color=colours[i], marker=i,
        #           label=list(results[i].keys())[i][0])
    plt.xlabel('Grid M')
    plt.ylabel('Absolute error of the price')
    plt.title('Log-log plot for Fourier fourier')
    plt.ylim(pow(10, -17), pow(10, 1))
    plt.legend(loc='upper center')
    plt.show()
    plt.close(0)
    return
