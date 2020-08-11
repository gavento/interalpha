
import tensorflow.keras as K
import utils
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load():
    sgf = utils.load_sgf("30va-gokifu-20191209-Mi_Yuting-Peng_Liyao.sgf")

def show_filters(layers):
    ws, bs = layers["init_conv_block"].get_weights()
    ix = 1
    rows = 26
    cols = [0, 8, 16, 17]
    for i in range(rows):
        # get the filter
        f = ws[:, :, :, i]
        # plot each channel separately
        for j in cols:
            # specify subplot and turn of axis
            ax = plt.subplot(rows, len(cols), ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

def show_layer(model, layer, inp):
    model2 = K.models.Model(inputs=model.inputs, outputs=[layer.output])
    output = model2.predict(inp)[0]
    print(output.shape)

    square = 8
    fig = make_subplots(rows=8, cols=8)
    # plot all 64 maps in an 8x8 squares
    ix = 0
    for i in range(square):
        for j in range(square):
            # specify subplot and turn of axis
            #ax = plt.subplot(square, square, ix + 1)
            #ax.set_xticks([])
            #ax.set_yticks([])
            # plot filter channel in grayscale
            #plt.imshow(output[:, :, ix], cmap='gray')
            #fig.add_imshow(output[:, :, ix], row=i, col=j)
            h = go.Heatmap(z=output[:, ::-1, ix], colorscale="gray")
            fig.add_trace(h, row=i + 1, col=j + 1)
            ix += 1
    # show the figure
    fig.show()


def main():
    inp = load()
    inp = np.expand_dims(inp, 0)
    print(inp.shape)
    model = K.models.load_model("leela.weights.h5")
    model.summary()

    layers = dict((layer.name, layer) for layer in model.layers)

    #name = "residual_2_0_conv_block"
    name = "residual_1_15_conv_block"
    #name = "init_conv_block"
    show_layer(model, layers[name], inp)

    return

    policy, value = model.predict(inp)
    policy = policy[0]
    value = value[0]

    print("PASS:", policy[-1])
    actions = policy[:-1].reshape((19, 19))
    plt.imshow(np.moveaxis(actions, 0, -1))
    plt.show()

if __name__ == "__main__":
    main()