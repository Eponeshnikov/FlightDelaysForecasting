import matplotlib.pyplot as plt


def vis_params_delay(data, max_cmap, *params, show=False, name_=''):
    """
    Function which plots figures. OX - params, OY - Delay
    :param data: data for visualizing
    :param max_cmap: reference point to save colormap scale
    :param params: params to plot
    :param show: show figures
    :param name_: the second part of the name of the figures
    :return: None
    """
    # add reference point to save colormap scale
    row = data.iloc[[len(data) - 1]]
    row['Delay'] = max_cmap + 10
    row.index.values[0] = len(data)
    data = data.append(row)
    name = ''

    # settings
    nrows, ncols = 3, round(len(params) / 3)  # array of sub-plots
    if nrows * ncols < len(params):
        ncols += 1
    figsize = [12, 7]  # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for axi, param in zip(ax.flat, params):
        axi.set_title(f"Delay depending on {param}")
        axi.set_xlabel(param)
        if param == 'Flight Duration':
            axi.set_xlabel(f"{param} (min)")
        axi.set_ylabel("Delay (min)")
        axi.set_ylim(0, max_cmap)
        axi.scatter(data[param], data['Delay'], marker='.', c=data['Delay'], cmap='rainbow')
        name += param

    plt.tight_layout(True)

    plt.savefig(f"plots/{name}_{name_}.jpg")
    if show:
        plt.show()
    # delete reference point
    data = data.drop(len(data) - 1)
