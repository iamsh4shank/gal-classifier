import itertools
import numpy as np
import matplotlib.pyplot as plt   

def plot_initial_eg(imageLabel,
                    labels,
                    images):
    fig, axes = plt.subplots(ncols = 10, nrows = 10, figsize = (20,20))
    index = 0
    for i in range(10):
        for j in range(10):
            axes[i,j].set_title(imageLabel[labels[index]])
            axes[i,j].imshow(images[index].astype(np.uint8))
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
            index +=1
    plt.show()

def plot_confusionM(cm, class_names):
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_final_viz(imageLabel, 
                   actual_label, 
                   pred_label, 
                   test_images):

    fig, axes = plt.subplots(ncols=7, nrows=3, sharex=False,
    sharey=True, figsize=(17, 8))
    index = 0
    for i in range(3):
        for j in range(7):
            axes[i,j].set_title('actual:' + imageLabel[actual_label[index]] + '\n' 
                                + 'predicted:' + imageLabel[pred_label[index]])
            axes[i,j].imshow(test_images[index].astype(np.uint8), cmap='gray')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
            index += 1
    plt.show()