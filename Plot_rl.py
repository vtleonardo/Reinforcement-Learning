import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from Base_agent import Agent, agent_arg_parser
import cv2
from PIL import Image
from keras import backend as K
import tensorflow as tf
import re

def plot_stats(path_csv_folder, csv_files, names_plot):
    """
    Function that plots the relevant information about the training. The following items are plotted:
            Average Q values
            Average Rewards
            Average Loss
            Performance in frames/second
            Average fps in the training
    :param path_csv_folder: string
            Path to the folder that contains the .cv files to be plotted.
    :param csv_files: list of strings
            The names of the .csv files to be plotted.
    :param names_plot: list of strings
            Names of the curves in the plot.
    :return: nothing
    """
    abs_path = path_csv_folder
    path_csv = csv_files
    name = names_plot
    # Plot variables
    colors = ["C{}".format(i) for i in range(10)]
    data_frame = []
    # moving average window
    opt_window = 50
    # 1 epoch is equal to this quantity
    epoch = 50000
    # Variable that controls the transparency of the real values in front of the mean values, float (0 to 1)
    alpha = 0.2
    mean_fps = []
    leg_d= []
    min_vect, max_vect = [], []
    # reading each csv
    for i, path in enumerate(path_csv):
        path_temp= os.path.join(abs_path, path)
        if (os.path.exists(path_temp)):
            dta = pd.read_csv(os.path.join(abs_path, path))
            data_frame.append(dta)
    for i, df in enumerate(data_frame):
        # moving average
        df_2 = df.rolling(window=opt_window, center=True).mean()
        #print(df.mean())
        tim = df["Time"].sum()
        print("Total number of frames seen in the training:{}".format(df["Num_frames"].max()))
        print("Training time:{:.2f} seconds".format(tim))
        print("Training time:{:.2f} hours".format(tim / 3600.0))
        plt.figure(1)
        if epoch == 1:
            plt.xlabel("Frames")
        else:
            plt.xlabel("Epochs (1 epoch = {} frames)".format(epoch))
        plt.ylabel("Average Q value")
        plt.plot(df["Num_frames"] / epoch, df["Q_value"], color=colors[i], alpha=alpha)
        aux,=plt.plot(df_2["Num_frames"] / epoch, df_2["Q_value"], color=colors[i], label="{}".format(name[i]))
        leg_d.append(aux)
        plt.legend(handles=leg_d)
        plt.grid()
        plt.title("Average value of the Q-function")

        plt.figure(2)
        plt.plot(df["Num_frames"] / epoch, df["Rewards"], color=colors[i], alpha=alpha)
        plt.plot(df_2["Num_frames"] / epoch, df_2["Rewards"], color=colors[i])
        if epoch == 1:
            plt.xlabel("Frames")
        else:
            plt.xlabel("Epochs (1 epoch = {} frames)".format(epoch))
        plt.ylabel("Average Reward")
        plt.legend(handles=leg_d)
        plt.xticks(np.arange(0,11,0.5))
        min_vect.append(df["Rewards"].min())
        min_vect.append(df_2["Rewards"].min())
        max_vect.append(df["Rewards"].max())
        max_vect.append(df_2["Rewards"].max())
        plt.gca().set_xlim(0,10)
        plt.gca().set_ylim(np.amin(min_vect),np.amax(max_vect)+5)
        #plt.grid()
        plt.title("Average rewards received")

        plt.figure(3)
        plt.plot(df["Num_frames"] / epoch, df["Loss"], color=colors[i], alpha=alpha)
        plt.plot(df_2["Num_frames"] / epoch, df_2["Loss"], color=colors[i])
        if epoch == 1:
            plt.xlabel("Frames")
        else:
            plt.xlabel("Epochs (1 epoch = {} frames)".format(epoch))
        plt.ylabel("Average Loss")
        plt.legend(handles=leg_d)
        plt.grid()
        plt.title("Average Loss")

        plt.figure(4)
        plt.plot(df["Num_frames"], df["FPS"], color=colors[i], alpha=alpha)
        plt.plot(df_2["Num_frames"], df_2["FPS"], color=colors[i])
        plt.xlabel("Frames")
        plt.ylabel("Frames/Second")
        plt.legend(handles=leg_d)
        plt.grid()
        plt.title("Performance in Frames/Second")
        mean_fps.append(df["FPS"].mean())
    plt.figure(5)
    plt.title("Average Frames/Second in the training")
    plt.bar(np.arange(len(mean_fps)), mean_fps, color=colors)
    for i,mean in enumerate(mean_fps):
        plt.text(x=(i)-0.1, y=mean/2.0, s="Average FPS:{:.2f}".format(mean))
    plt.ylabel("Frames/Second")
    plt.xticks(np.arange(len(mean_fps)),name)
    plt.yticks()
    plt.figure(1)
    plt.grid()
    plt.figure(2)
    plt.grid()
    plt.figure(3)
    plt.grid()
    plt.figure(4)
    plt.grid()
    plt.show()


def get_image_max_filter(model_input, layer_output, num_filters, input_shape, lr=2, iterations=1000,
                         input_img_data=np.array([])):
    """
    Function that finds the input that maximize the each filters' activation in a layers of the
    desired network. Based on the tutorial:
    How convolutional neural networks see the world. By Francois Chollet.
    https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

    :param model_input: Tensor.input
            Model's input as a tensor
    :param layer_output: Tensor.output
            The desired layer's output that contains the filters to be maximized
    :param num_filters: int
            Number of layer's filters to be maximized
    :param input_shape: tuple of int (Width, height, Depth)
            Input volume's shape (state shape) that is fed to the model.
    :param lr:
            Training's Learning rate.
    :param iterations:
            Number of iterations that the algorithm will be trained.
    :return: tuple with:
            list of input_img_data : np.array (dtype=np.uint8)
                Images that maximizes each of the filters in a layer.
            list of loss_vect : np.array float32
                History with the loss along the "training"
    """
    print("Starting the training of the inputs that maximizes the filter of the conv layer:{}".
          format(layer_output.name))
    if input_img_data.shape[0] == 0:
        input_img_data = []
        for i in range(num_filters):
            # we start from image with some noise
            input_img_data.append(np.random.random((1, *input_shape)) * 20 + 128.)
    input_img_data = np.concatenate(input_img_data, axis=0)
    # compute the gradient of the input picture wrt this loss
    loss = [tf.reduce_mean(layer_output[i, :, :, i]) for i in range(num_filters)]
    joint_loss = tf.reduce_sum(loss)
    grads = tf.gradients([joint_loss], [model_input])
    # Manually defining the Adam optimizer (default parameters)
    beta1 = tf.constant(0.9)
    beta2 = tf.constant(0.99)
    eps = tf.constant(1e-8)
    t = tf.placeholder(dtype=tf.float32)
    m = tf.constant(0.0)
    v = tf.constant(0.0)
    m = beta1 * m + (1 - beta1) * grads
    mt = m / (1 - tf.pow(beta1, t))
    v = beta2 * v + (1 - beta2) * (tf.pow(grads, 2))
    vt = v / (1 - tf.pow(beta2, t))
    grads_adam = mt / (tf.sqrt(vt) + eps)
    # Starting the optimization
    loss_vect = [[] for i in range(num_filters)]
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        for i in range(iterations):
            print("Iteration:{}/{}".format(i,iterations))
            grad_value, loss_aux = sess.run([grads_adam, loss], feed_dict={model_input: input_img_data,
                                                                           t: (i + 1)})
            input_img_data += np.clip(np.round(grad_value[0] * lr), -10, 10)
            input_img_data = np.clip(input_img_data,0,255)
            # input_img_data = deprocess_image(input_img_data).astype(dtype=np.float32)
            for i in range(num_filters):
                loss_vect[i].append(loss_aux[i])

    img_list = [deprocess_image(np.squeeze(input_img_data[i])) for i in range(input_img_data.shape[0])]
    return (img_list, loss_vect)


def deprocess_image(img):
    """
    Function that normalize a image/tensor of dtype=float and convert to a RGB image
    dtype=np.uint8 with range of [0,255].

    :param img: np.array(dtype=float)
    :return: img : np.array(dtype=np.uint8)
    """
    # normalize tensor: center on 0., ensure std is 0.1
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    # clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)
    # convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')

    return img

def get_rows_cols(num_elements):
    """
    Function that finds the best number of rows and columns given the number of elements.
    :param num_elements: int
            Number of elements.
    :return: tuple : int (rows, columns)
    """
    rows, cols = 0, 0
    # Exception to the rule
    if num_elements == 32:
        rows, cols = 8, 4
    # Trying to let rows and columns the closest possible
    else:
        cols = np.ceil(np.sqrt(num_elements)).astype(np.int)
        for i in range(cols,-1,-1):
            if cols * i < num_elements:
                rows = i+1
                break
    return (rows,cols)

def join_image(img ,input_depth=1, width_space=1, height_space=1, channel_last=True):
    """
    Function that join several images in one.
    :param img:
            Image volumes to be joined with shape (Width, Height, Depth*number of elements) or
            (Number of elements, Width, Height, Depth)
    :param input_depth: int (Default:1 ie. gray)
            Number of the color's channel in the input image. ie (gray = 1 RGB = 3)
    :param width_space: int (Default:1 pixel)
            Horizontal space in pixels between each image in the total image.
    :param height_space: int (Default:1 pixel)
            Vertical space in pixels between each image in the total image.
    :param channel_last: bool (Default:True)
            If the number of elements is in the first or last axis. If true the number of elements is in
            the last axis.
    :return: Resulting image (np.array of dtype=uint8)
    """
    # Reshaping the input volume putting the number of frames as the first axis.
    if len(img.shape) != 4 and channel_last:
        img_aux=[np.expand_dims(img[:,:,i:i+input_depth],axis=0) for i in range(0,img.shape[2],input_depth)]
        img = np.concatenate(img_aux,axis=0)
    mode = "L" if input_depth == 1 else "RGB"
    num_elements, img_width, img_height = img.shape[0], img.shape[1], img.shape[2]
    rows, cols = get_rows_cols(num_elements)
    total_width = (cols * (img_width + width_space)) + width_space  # +width_space for the last space
    total_height = (rows * (img_height + height_space)) + height_space
    new_img = Image.new(mode=mode, size=(total_width, total_height))
    for i in range(num_elements):
        idx, idy = np.unravel_index(i, (cols, rows), order="F")
        x_offset = (idx * (img_width + width_space)) + width_space
        y_offset = (idy * (img_height + height_space)) + height_space
        if img[i].dtype == np.float32:
            new_img_aux = Image.fromarray(np.squeeze(deprocess_image(img[i])), mode=mode)
        else:
            new_img_aux = Image.fromarray(np.squeeze(img[i]), mode=mode)
        new_img.paste(new_img_aux, (x_offset, y_offset))
    return np.array(new_img)

def deconv_coord(conv_layer,layer_number,coord_x,coord_y):
    """
    Function that computes the rectangle's size and coordinates in the input image that corresponds to
    local that activated the maximum activation in a layer.
    :param conv_layer: list of tensors
            List of the model's convolution layers.
    :param layer_number: int
            The number of the layer to be analysed.
    :param coord_x: int
            X coordinate of the activation to be analysed.
    :param coord_y: int
            Y coordinate of the activation to be analysed.
    :return: tuple with:
            coord_x, coord_y : Top left coordinates of the rectangle.
            rect_width and rect_height : Width and Height of the rectangle in the input image.
    """
    rect_width = 0
    rect_height = 0
    first_time = True
    # Calculating the rectangle's size and coordinates in the input image (deconvolving)
    for j in range(layer_number, -1, -1):
        strides = conv_layer[j].strides
        kernel_size = conv_layer[j].kernel_size
        # Calculating the padding
        W_1 = conv_layer[j].input_shape[1]
        W_2 = conv_layer[j].output_shape[1]
        padding = (W_2 * strides[0] + kernel_size[0] - strides[0] - W_1) / 2
        # Calculating the coordinates on the input image
        coord_x = int(coord_x * strides[0] - padding)
        coord_y = int(coord_y * strides[1] - padding)
        # size[-1] = F for the last layer, and size[-2]=(size[-1]-1)S + F1 for the previous ones to the last
        if first_time:
            first_time = False
            rect_width = kernel_size[0]
            rect_height = kernel_size[1]
        else:
            rect_width = (rect_width - 1) * strides[0] + kernel_size[0]
            rect_height = (rect_height - 1) * strides[1] + kernel_size[1]

    return (coord_x, coord_y, rect_width, rect_height)

def blend_img(frames, alpha=1.0, beta=0.6):
    """
    Function that blend a sequence of frames (state).
    :param frames: np.array
            Frames to be blended.
    :param alpha:
            Transparency of the newer frame in the sequence.
    :param beta:
            Transparency of the older frame in the sequence.
    :return: the resulting image (np.array)
    """
    bld_img = frames[0]
    for i in range(len(frames)-1):
        bld_img = cv2.addWeighted(frames[i+1],alpha,bld_img,beta,0)
    return bld_img

def get_convolutional_layers(agent):
    """
    Function that finds each convolution layer in the model.
    :param agent:
            RL agent that possess the neural network model.
    :return: list of tensors with each convolution layer.
    """
    conv_layer = []
    conv_eval = []
    check_for_input = True
    conv_inputs = []
    for i in range(len(agent.Q_value.layers)):
        #Checks for all inputs that come before the first conv layer.
        if check_for_input and "input" in agent.Q_value.layers[i].__str__():
            conv_inputs.append(agent.Q_value.layers[i].input)
        if "conv" in agent.Q_value.layers[i].__str__():
            check_for_input = False
            conv_layer.append(agent.Q_value.layers[i])
            conv_eval.append(K.function(conv_inputs,[agent.Q_value.layers[i].output]))
    return (conv_layer,conv_eval)

def plot_network(agent, state_path, state_save, n_maximum=[20,3,3]):
    """
    Function that plots the relevant information about the agent's network. These information are:
            Input State(sequence of frames) as a blended image.
            The activation maps of each layer
            The parts in the input image that trigger the N maximum activations of a convolutional layer for
            each layer in the model.
            The input images (and its losses) that maximizes each of the filters in a layer for each
            convolutional layer in the model.
            The first convolutional layer's weights.
            The Action-values (Q) for each action available for the input state.
    :param agent: RL agent
            RL agent that contains the model to be analysed
    :param state_path: string
            Path to file as .gif that contains the state to be analysed.
    :param state_save: string
            Path to the folder where the resulting images will be saved.
    :param n_maximum: list of ints
            List that contains the number os maximum activation that will be analysed in each conv layer.Hence,
            the total number of elements should be equal to the number of convolutional layer in the model.
    :return: nothing
    """
    if agent.input_depth ==3:
        cmap = None
        cmap2 = None
    else:
        cmap = "gray_r"
        cmap2 = "gray"
    fig_idx = 0
    state=Image.open(state_path)
    frames= []
    for frame in range(0,state.n_frames):
        state.seek(frame)
        state_aux = state.convert("RGB") if agent.input_depth == 3 else state
        frames.append(np.array(state_aux.copy().getdata(),dtype=np.uint8).reshape(agent.input_shape))
    bld_img=blend_img(frames)
    Image.fromarray(bld_img).save(os.path.join(state_save, "blended2_img.png"))
    fig0 = plt.figure(fig_idx)
    fig_idx += 1
    plt.imshow(bld_img.astype(np.uint8), cmap=cmap)
    plt.imsave(os.path.join(state_save, "blended_img.png"), bld_img, cmap=cmap)
    for i in range(len(frames)):
        plt.imsave(os.path.join(state_save, "frame{}.png".format(i)), np.squeeze(frames[i]), cmap=cmap)
    #=====================================================================================================#
    #    Gets each layer's activation and the portions of the input image that maximize its filters.
    #=====================================================================================================#
    if agent.is_recurrent:
        state_vect = np.concatenate(np.expand_dims(frames, axis=0), axis=0)
    else:
        state_vect = np.expand_dims(np.concatenate(frames, axis=2), axis=0)
    conv_layer,conv_eval = get_convolutional_layers(agent)
    for i in range(len(conv_layer)):
        img = np.squeeze(conv_eval[i]([state_vect])[0])
        num_elements = img.shape[2]
        fig1=plt.figure(fig_idx,figsize=(8,8))
        fig_idx+=1
        fig1.suptitle("Convolutional layer {}: Activations".format(i+1),fontsize=16)
        new_img = join_image(img)
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.axis("off")
        plt_save=plt.imshow(np.asarray(new_img,dtype=np.float32),cmap=cmap2)._A
        plt.imsave(os.path.join(state_save, "activ-layer{}.png".format(i)),plt_save , cmap=cmap2)
        # Store the argmax, max and index of each filter
        argmax_vect = []
        max_vect = []
        filter_idx = []
        n_max = n_maximum
        # Case n_max has less elements than the number of layers
        while len(n_max) < len(conv_layer): n_max.append(1)
        for j in range(num_elements):
            img_cpy = img[:, :, j].copy()
            # Get n maximum elements of a filter
            # finds the first max and replace it for zero and repeat until it catch the n elements)
            for k in range(n_max[i]):
                arg_max = np.unravel_index(np.argmax(img_cpy), img_cpy.shape, order="C")
                argmax_vect.append(arg_max)
                max_vect.append(np.amax(img_cpy))
                img_cpy[arg_max] = 0.0
                filter_idx.append(j)
        fig1 = plt.figure(fig_idx, figsize=(8, 8))
        fig_idx += 1
        ax = plt.gca()
        filter_idx_max = []
        # Plotting on the input the rectangles that maximizes the activations of this layer.
        for k in range(n_max[i]):
            idx_max = int(np.argmax(max_vect))
            # Images uses (x,y) arrays uses (y,x)[rows,cols]
            ret_top_left = (argmax_vect[idx_max][1],argmax_vect[idx_max][0])
            max_vect[idx_max] = 0.0
            filter_idx_max.append(filter_idx[idx_max])
            coord_x, coord_y = ret_top_left[0] , ret_top_left[1]
            coord_x,coord_y,rect_width,rect_height = deconv_coord(conv_layer=conv_layer,layer_number=i,
                                                                  coord_x=coord_x,coord_y=coord_y)
            rect=patches.Rectangle((coord_x, coord_y), rect_width, rect_height, linewidth=1,
                                   edgecolor="red", facecolor='none',alpha=0.95)
            ax.add_patch(rect)

        plt_save=ax.imshow(bld_img, cmap=cmap)._A
        plt.imsave(os.path.join(state_save, "max-zone-layer{}.png".format(i)), plt_save, cmap=cmap)
        # ====================================================================================#
        #   Gets the input image that maximize each of the network`s filters
        # ====================================================================================#
        # Taking out the repeated elements
        for i_aux in range(len(filter_idx_max)):
            for j in range(i_aux,len(filter_idx_max)-1):
                if filter_idx_max[i_aux] == -1:
                    break
                if filter_idx_max[i_aux]==filter_idx_max[j+1]:
                    filter_idx_max[j+1]=-1
        filter_idx_max = [idx for idx in filter_idx_max if idx!=-1]
        n_idx = len(filter_idx_max)
        fig1 = plt.figure(fig_idx, figsize=(8, 8))
        fig1.suptitle("Losses dos filtros da camada convolutiva {}".format(i+1))
        fig_idx += 1
        rows,cols = get_rows_cols(num_elements)
        axes = [fig1.add_subplot(rows,cols,k+1) for k in range(num_elements)]
        img_max = []
        img_temp, loss = get_image_max_filter(model_input=agent.Q_value.layers[0].input,
                                              layer_output=conv_layer[i].output, num_filters=num_elements,
                                              input_shape=agent.state_input_shape)
        for j in range(num_elements):
            img_max.append(np.expand_dims(np.asarray(join_image(img_temp[j], input_depth=agent.input_depth)),
                                          axis=0))
            axes[j].plot(np.arange(len(loss[j])),np.array(loss[j]))
            axes[j].set_xticks([])
            axes[j].set_yticks([])
        fig1 = plt.figure(fig_idx, figsize=(8, 8))
        fig_idx += 1
        img_total = np.asarray(join_image(np.concatenate(img_max,axis=0),
                            input_depth=agent.input_depth,width_space=3,height_space=3, channel_last=False))
        im=plt.imshow(img_total,cmap=cmap)._A
        plt.imsave(os.path.join(state_save, "filters-layer{}.png".format(i)), im, cmap=cmap)

    #=====================================================================#
    #   Displaying the first layer's filters
    #=====================================================================#
    weights=agent.Q_value.get_weights()[0]
    n_filters = weights.shape[-1]
    fig2 = plt.figure(fig_idx,figsize=(4, 4))
    fig_idx +=1
    fig2.suptitle("First Convolutional Layer's filters", fontsize=16)
    filters_vec = []
    # joing each slice of a filter in one image
    for i in range(n_filters):
        filters_vec.append(np.expand_dims(join_image(np.squeeze(weights[:,:,:,i]),
                                                     input_depth=agent.input_depth),axis=0))
    # joining all the filters
    filter_total = np.squeeze(join_image(np.concatenate(filters_vec,axis=0),width_space=2,height_space=2,
                                         input_depth=agent.input_depth, channel_last=False))
    plt_save=plt.imshow(filter_total,cmap="gray")._A
    plt.imsave(os.path.join(state_save, "layer0-filters.png"), plt_save, cmap=cmap2)
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    ax.axis("off")
    #==================================================================================================#
    #   Bar plot with the q-values
    #==================================================================================================#
    q_s = agent.Q_value.predict_on_batch([state_vect])
    fig2 = plt.figure(fig_idx, figsize=(4, 4))
    fig2.suptitle("Action-Values (Q) for each action available")
    fig_idx += 1
    coordx=np.arange(q_s.size)*0.8
    colors=["C{}".format(i) for i in range(q_s.size)]
    plt.bar(x=coordx,height=np.squeeze(q_s),color=colors)
    plt.xticks(coordx,agent.env.action_meanings())
    plt.ylabel("Action-Values (Q)")
    plt.xlabel("Actions (Buttons)")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the RL-variables.")
    parser.add_argument("--plot_mode", choices=["stats","network"], default="stats",
        help="Mode to execute the plot. Type:str. Default=stats")
    parser.add_argument("--state", default="",
        help="Path to .gif file that contains the state to be plotted. Default: None. "
             "REQUIRED IN PLOT NETWORK MODE")
    parser.add_argument("--path_state_save", default="",
        help="Path to the folder where the results will be saved on the disk. Default: None. "
             "REQUIRED IN PLOT NETWORK MODE")
    parser.add_argument("--n_maximum", default="",
        help="List with the number of maximum activation to be analysed in each convolutional layer. Hence,"
             "the total number of elements should be equal to the number of convolutional layer in the model"
             "Type:str (with each argument separated by space or comma, and the whole sentence between "
             "quotation marks). Default:\"84 84\"")
    parser.add_argument("--path_csv_folder", default="",
         help="Path to folder that contains the .csv to be plotted Default: None."
            "REQUIRED IN THE STATS MODE.")
    parser.add_argument("--csv_files", default="",
        help="List with the names of .csv files separated by \',\' and between quotation marks that contains "
             "the training information to be plotted. Default: None. Eg.\"my_file1.csv,my_file2.csv...\" "
             "REQUIRED IN THE STATS MODE." )
    parser.add_argument("--names_plot", default="",
        help="List with the names of each .csv file to be plotted separated by \',\' and between quotation "
             "marks. Default: None. Eg.\"name_csv1, name_csv2...\"  REQUIRED IN THE STATS MODE")
    args, kwargs = agent_arg_parser(parser)
    kwargs["silent_mode"] = True
    kwargs["load_weights"] = True
    if args.plot_mode.lower() == "stats":
        csv_files_aux = args.csv_files.replace("\"","").split(",")
        names_plot_aux = args.names_plot.replace("\"","").split(",")
        plot_stats(path_csv_folder=args.path_csv_folder,csv_files=csv_files_aux, names_plot=names_plot_aux)
    else:
        if args.weights_load_path == "":
            raise Exception("The path to load the weights was not valid!")
        agent = Agent(**kwargs)
        n_maximum_aux = [int(item) for item in re.findall(r"\d+", args.n_maximum)]
        plot_network(agent=agent, state_path=args.state, state_save=args.path_state_save,
                     n_maximum=n_maximum_aux)
