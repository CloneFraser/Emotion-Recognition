import matplotlib.pyplot as plt


class plot_model:
    def __init__(self, model_history):
        self.plot_model_history(model_history)


    def plot_model_history(self, model_history):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # summarize history for accuracy
        axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
        axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
        axs[0].legend(['train', 'val'], loc='best')

        # summarize history for loss
        axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
        axs[1].legend(['train', 'val'], loc='best')
        fig.savefig('plot.png')
        plt.show()