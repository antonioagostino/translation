from matplotlib.pylab import plt
import numpy as np

def plot_loss(train_loss_filepath: str,
              val_loss_filepath: str,
              output_filepath: str) -> None:
    """
    It imports the history of training and validation losses, and
    plots the losses using Matplotlib. It saves to a PNG file the plot.

    Args:
        train_loss_filepath (str): Filepath of the training loss log.
        val_loss_filepath (str): Filepath of the validation loss log.
        output_filepath (str): Where to save the plot as PNG image.
    """
    train_loss = None
    train_loss_dict = {}
    with open("training_losses.txt", "r") as train_file:
        train_loss = train_file.readlines()

    for line in train_loss:
        iteration, loss = line.split(":", 1)
        train_loss_dict[int(iteration) - 500] = float(loss)

    val_loss = None
    val_loss_dict = {}
    with open("val_losses.txt", "r") as val_file:
        val_loss = val_file.readlines()

    for line in val_loss:
        iteration, loss = line.split(":", 1)
        val_loss_dict[int(iteration) - 500] = float(loss)
    

    plt.plot(list(train_loss_dict.keys()), list(train_loss_dict.values()), label='Training Loss')
    plt.plot(list(val_loss_dict.keys()), list(val_loss_dict.values()), label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # We take 1/8 of the iterations
    plt.xticks(np.array(list(train_loss_dict.keys())[::8]))
    
    # Display the plot
    plt.legend(loc='best')
    plt.savefig("loss.png")