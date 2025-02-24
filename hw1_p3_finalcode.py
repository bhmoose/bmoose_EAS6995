import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

#Found this solution online as workaround to OpenMP runtime issue
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load the MNIST dataset 
def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])  # Only convert to tensor

    # Download the dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # train_dataset.data = train_dataset.data / 255
    # test_dataset.data = test_dataset.data / 255

    # Subsampling: 50% from each class
    train_indices = subsample_50_percent_per_class(train_dataset)
    train_subset = Subset(train_dataset, train_indices)

    test_indices = subsample_50_percent_per_class(test_dataset)
    test_subset = Subset(test_dataset, test_indices)

    # DataLoader for batching
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to perform subsampling 50% from each class
def subsample_50_percent_per_class(dataset):
    """
    Subsample 50% of the data from each class.
    dataset: The full dataset 
    Returns: A list of indices for the subsampled dataset
    """
    # --- DONE: Implement subsampling logic here ---

    #Get all labels in dataset
    all_labels = np.array([dataset[i][1] for i in range(dataset.data.shape[0])])

    sampled_indices = []
    #Iterate through classes/labels
    for j in range(10):
        #Find indices where label is a certain value
        idx_array = np.where(all_labels == j)[0]
        #Keep 50 percent of these samples, add to sampled_indices list
        random_idxs_class = np.random.randint(0, len(idx_array), size = int(np.round(len(idx_array)/2)))
        sampled_indices.extend(idx_array[random_idxs_class])

    return sampled_indices


# Forward pass for Fully Connected Layer
def fully_connected_forward(X, W, b):
    """
    Perform forward pass for a fully connected (linear) layer.
    X: Input data
    W: Weight matrix
    b: Bias vector
    Returns: Z, the output of the linear layer
    """
    Z = (X@W) + b   #Compute the linear transformation (X * W + b)
    return Z

# Forward pass for ReLU activation
def relu_forward(Z):
    """
    ReLU activation function forward pass.
    Z: Linear output (input to ReLU)
    Returns: A, the output of the ReLU function applied elementwise to Z
    """
    A = np.where(Z < 0, 0, Z)  #Apply ReLU function (element-wise)
    return A

# Forward pass for Softmax activation
def softmax_forward(Z):
    """
    Softmax activation function forward pass.
    Z: Output logits (before softmax)
    Returns: output, the output of the softmax function applied to Z (same dimensions as Z)
    """
    exp_z = np.exp(Z)  # DONE: Apply softmax function (numerical stability)
    output = exp_z / np.sum(exp_z, axis = 1).reshape([Z.shape[0], 1])  # DONE: Normalize exp_z to get the softmax output
    return output

# Backward pass for Fully Connected Layer (Linear)
def fully_connected_backward(X, W, dZ):
    """
    Compute gradients for the fully connected (linear) layer.
    X: Input data (Nxd)
    W: Weight matrix (dxK)
    dZ: Gradient of the loss with respect to Z (from the next layer)
    Returns:
    dW, the derivative of the loss with respect to the weights W
    db, the derivative of the loss with respect to the bias b
    dA, the derivative of the loss with respect to the output of the previous layer's activation function
    """
    #Compute gradient of weights (X^T * dZ)
    dW = np.transpose(X)@dZ  
    #Compute gradient of bias
    db = (1/dZ.shape[0]) * np.sum(dZ, axis = 0)  
    #Compute gradient of loss with respect to A (for backpropagation)
    dA = dZ@np.transpose(W)
    return dW, db, dA

# Backward pass for ReLU activation
def relu_backward(Z, dA):
    """
    Compute the gradient for ReLU activation.
    Z: Input to ReLU (before activation)
    dA: Gradient of the loss with respect to activations (from the next layer)
    Returns: dZ, the derivative of the loss with respect to the ReLU input Z
    """
    dZ = np.where(Z <= 0, 0, dA)  # Compute dZ for ReLU (gradient is 0 for Z <= 0 and dA for Z > 0)
    return dZ

# Backward pass for Softmax Layer
def softmax_backward(S, Y):
    """
    NOTE THE CORRECTION/EFFICIENCY GAIN HERE in using softmax output instead of Z
    Compute the gradient of the loss with respect to softmax output.
    S: Output of softmax 
    Y: True labels (one-hot encoded)
    Returns: dZ, the derivative of the loss with respect to the softmax input (accounts for both softmax and cross-entropy derivatives)
    """
    
    all_samples_vec = []
    #Iterate through samples in batch
    for k in range(S.shape[0]):
        single_sample_vec = []
        #Iterate through logits in sample (output of softmax)
        for j in range(S.shape[1]):
            all_dL_dzj = []
            #Iterate through classes
            for i in range(S.shape[1]):
                #Get derivative of cross-entropy loss with respect to softmax output a_i
                dL_dai = S[k, i] - Y[k, i]
                #If classes are the same, compute (d(a_i)/d(z_j))
                if i == j:
                    dai_dzj = S[k, i] * (1 - S[k, i])
                #If classes are different, compute (d(a_i)/d(z_j))
                else:
                    dai_dzj = -1 * S[k, i] * S[k, j]
                #Calculate (dL/d(z_j)) via the intermediate step of a_i
                all_dL_dzj.append(dL_dai * dai_dzj)
            #Sum all entries in all_dL_dzj to get dL/d(z_j)
            #Append each dL/d(z_j) -- for different j -- to list
            single_sample_vec.append(sum(all_dL_dzj))
        #Append result for single sample to list containing all samples' results
        all_samples_vec.append(single_sample_vec)
    dZ = np.array(all_samples_vec) 
    
    return dZ

# Weight update function (gradient descent)
def update_weights(weights, biases, grads_W, grads_b, learning_rate):
    """
    --- DONE: Implement the weight update step ---
    weights: Current weights
    biases: Current biases
    grads_W: Gradient of the weights
    grads_b: Gradient of the biases
    learning_rate: Learning rate for gradient descent
    """
    #Update weights and biases with product of learning rate and gradient of loss function
    new_weights = weights - (learning_rate * grads_W)
    new_biases = biases - (learning_rate * grads_b)

    return new_weights, new_biases


# Define the neural network 
def train(train_loader, test_loader, epochs=10000, learning_rate=0.001):
    # Initialize weights and biases
    input_dim = 784     
    hidden_dim1 = 128   #could set differently
    hidden_dim2 = 64    #could set differently
    output_dim = 10     
    
    # Initialize weights randomly
    W1 = np.random.randn(input_dim, hidden_dim1) * 0.01 
    b1 = np.zeros(hidden_dim1)
    W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01  # DONE
    b2 = np.zeros(hidden_dim2)                             # DONE
    W3 = np.random.randn(hidden_dim2, output_dim) * 0.01   # DONE
    b3 = np.zeros(output_dim)                              # DONE

    # ADD THESE to save training and test loss, accuracy
    training_loss = []
    test_loss = []
    training_accuracy = []
    test_accuracy = []
    
    # Loop through epochs
    for epoch in range(epochs):
        epoch_loss = 0
        test_epoch_loss = 0
        correct_predictions = 0
        total_correct_predictions = 0
        total_samples = 0

        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):

            # Flatten images to vectors
            actual_batch_size_train = X_batch.shape[0]
            X_batch = X_batch.squeeze().reshape([actual_batch_size_train, 28*28]) # Flatten images to vector
            Y_batch = torch.eye(output_dim)[Y_batch]  # Map label indices to corresponding one-hot encoded vectors

            # CONVERT TORCH TENSORS to numpy
            X = X_batch.numpy() # DONE
            Y = Y_batch.numpy() # DONE
            
            # --- DONE: Implement the forward pass ---
            Z1 = fully_connected_forward(X, W1, b1)
            A1 = relu_forward(Z1)
            Z2 = fully_connected_forward(A1, W2, b2)
            A2 = relu_forward(Z2)
            Z3 = fully_connected_forward(A2, W3, b3)
            Y_prob = softmax_forward(Z3)
            Y_pred = np.argmax(Y_prob, axis=1)
            # --- 
            
            # --- DONE: Implement loss computation --
            #Compute cross entropy loss
            loss_vector = np.array([(-1) * np.log(Y_prob[i, np.argmax(Y[i])]) for i in range(Y_prob.shape[0])])
            #print(loss_vector)
            #Divide by batch size
            loss = (1/actual_batch_size_train) * np.sum(loss_vector)
            #print(loss)

            epoch_loss = epoch_loss + loss

            # --- DONE: Implement backward pass ---
            dZ3 = softmax_backward(Y_prob, Y)
            dW3, db3, dA2 = fully_connected_backward(A2, W3, dZ3)
            dZ2 = relu_backward(Z2, dA2)
            dW2, db2, dA1 = fully_connected_backward(A1, W2, dZ2)
            dZ1 = relu_backward(Z1, dA1)
            dW1, db1, dX = fully_connected_backward(X, W1, dZ1)

            # --- DONE: Implement weight update ---
            W1, b1 = update_weights(W1, b1, dW1, db1, learning_rate)
            W2, b2 = update_weights(W2, b2, dW2, db2, learning_rate)
            W3, b3 = update_weights(W3, b3, dW3, db3, learning_rate)

            # --- DONE: Track accuracy
            correct_predictions = np.sum(Y_pred == np.argmax(Y, axis=1)) #for this batch
            total_correct_predictions = total_correct_predictions + correct_predictions #for the entire epoch
            total_samples = total_samples + actual_batch_size_train #for entire epoch

            #print("Done with Batch " + str(batch_idx))

        # Print out the progress - CLARIFIED
        train_accuracy_epoch = total_correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {train_accuracy_epoch * 100}%")
        
        # Save the training loss and accuracy for each epoch to plot later
        training_loss.append(epoch_loss / len(train_loader))
        training_accuracy.append(train_accuracy_epoch)
        
        # Save the test loss and accuracy for every epoch to plot later
        # if epoch % 10 == 0:
        test_correct_predictions = 0
        test_total_correct_predictions = 0
        test_total_samples = 0

        for test_batch_idx, (X_test_batch, Y_test_batch) in enumerate(test_loader):

            # Flatten images to vectors
            actual_batch_size_test = X_test_batch.shape[0]
            X_test_batch = X_test_batch.squeeze().reshape([actual_batch_size_test, 28*28]) 
            # Map label indices to corresponding one-hot encoded vectors
            Y_test_batch = torch.eye(output_dim)[Y_test_batch]  
            #Convert torch tensors into numpy
            X_test = X_test_batch.numpy() 
            Y_test = Y_test_batch.numpy() 

            #Run the model on test data
            Z1_test = fully_connected_forward(X_test, W1, b1)
            A1_test = relu_forward(Z1_test)
            Z2_test = fully_connected_forward(A1_test, W2, b2)
            A2_test = relu_forward(Z2_test)
            Z3_test = fully_connected_forward(A2_test, W3, b3)
            Y_prob_test = softmax_forward(Z3_test)
            Y_pred_test = np.argmax(Y_prob_test, axis=1)

            #Compute test loss the same way as training loss, but with test predictions and output
            test_batch_loss_vector = np.array([(-1) * np.log(Y_prob_test[i, np.argmax(Y_test[i])]) for i in range(Y_prob_test.shape[0])])
            test_batch_loss = (1/actual_batch_size_test) * np.sum(test_batch_loss_vector)

            test_epoch_loss = test_epoch_loss + test_batch_loss

            test_correct_predictions = np.sum(Y_pred_test == np.argmax(Y_test, axis=1)) #for this batch
            test_total_correct_predictions = test_total_correct_predictions + test_correct_predictions #for the entire epoch
            test_total_samples = test_total_samples + actual_batch_size_test #for entire epoch

        # Print out the progress - CLARIFIED
        test_accuracy_epoch = test_total_correct_predictions / test_total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_epoch_loss / len(test_loader)}, Test Accuracy: {test_accuracy_epoch * 100}%")

        #Store test loss and accuracy
        test_loss.append(test_epoch_loss / len(test_loader))
        test_accuracy.append(test_accuracy_epoch)
        
    return training_loss, training_accuracy, test_loss, test_accuracy, W1, W2, W3, b1, b2, b3


def plot_corr_incorr_testdata(W1, W2, W3, b1, b2, b3, test_loader):
    """
    Plots sample of 8 correct and incorrectly classified images
    Inputs W1, W2, W3, b1, b2, b3: weights and biases from trained model
    Inputs test_loeader: loader for test data
    Returns nothing.
    """

    #Set up arrays and lists for correct and incorrect labels and images
    #(First entries in arrays / lists ignored here) 
    correct_images = np.zeros((1, 28*28))
    correct_labels = [-999]
    incorrect_images = np.zeros((1, 28*28))
    incorrect_labels = [-999]
    incorrect_label_reallabels = [-999]

    #Iterate through test data batches
    for test_batch_idx, (X_test_batch, Y_test_batch_no_onehot) in enumerate(test_loader):

        output_dim = 10
        # Flatten images to vectors
        actual_batch_size_test = X_test_batch.shape[0]
        X_test_batch = X_test_batch.squeeze().reshape([actual_batch_size_test, 28*28]) 
        # Map label indices to corresponding one-hot encoded vectors
        Y_test_batch = torch.eye(output_dim)[Y_test_batch_no_onehot]  
        #Convert torch tensors into numpy
        X_test = X_test_batch.numpy() 
        Y_test = Y_test_batch.numpy() 

        #Run model on test data batch
        Z1_test = fully_connected_forward(X_test, W1, b1)
        A1_test = relu_forward(Z1_test)
        Z2_test = fully_connected_forward(A1_test, W2, b2)
        A2_test = relu_forward(Z2_test)
        Z3_test = fully_connected_forward(A2_test, W3, b3)
        Y_prob_test = softmax_forward(Z3_test)
        Y_pred_test = np.argmax(Y_prob_test, axis=1)

        #Get correctly classified images
        correct_images = np.append(correct_images, (X_test[np.nonzero(Y_pred_test == np.argmax(Y_test, axis=1))[0], :]), axis = 0)
        #Get correctly classified image labels
        correct_labels.extend(list(Y_pred_test[np.nonzero(Y_pred_test == np.argmax(Y_test, axis=1))[0]]))

        #Get incorrectly classified images
        incorrect_images = np.append(incorrect_images, (X_test[np.nonzero(Y_pred_test != np.argmax(Y_test, axis=1))[0], :]), axis = 0)
        #Get incorrectly classified image labels
        incorrect_labels.extend(list(Y_pred_test[np.nonzero(Y_pred_test != np.argmax(Y_test, axis=1))[0]]))
        #Get real labels for incorrectly classified images
        Y_test_argmax = np.argmax(Y_test, axis=1)
        incorrect_label_reallabels.extend(list(Y_test_argmax[np.nonzero(Y_pred_test != np.argmax(Y_test, axis=1))[0]]))

    #Print number of correctly and incorrectly classified images
    print("There are " + str(len(correct_labels)) + " correct labels in the test dataset")
    print("There are " + str(len(incorrect_labels)) + " incorrect labels in the test dataset")

    #Get 8 random indices of correctly and incorrectly classified images
    rand_corr_indices = np.random.randint(0, len(correct_labels), 8)
    rand_incorr_indices = np.random.randint(0, len(incorrect_labels), 8)

    #Get images and labels associated with randomly selected indices
    rand_corr_images = correct_images[rand_corr_indices, :].reshape(8, 28, 28)
    rand_corr_labels = np.array(correct_labels)[rand_corr_indices]
    rand_incorr_images = incorrect_images[rand_incorr_indices, :].reshape(8, 28, 28)
    rand_incorr_labels = np.array(incorrect_labels)[rand_incorr_indices]
    rand_incorr_reallabels = np.array(incorrect_label_reallabels)[rand_incorr_indices]

    fig, ax = plt.subplots(2, 8, figsize = [15, 8])
    ax = ax.flatten()
    for i in range(8):
        im = ax[i].matshow(rand_corr_images[i], cmap = 'Grays', vmin = 0, vmax = 1)
        ax[i].set_title("Label: " + str(rand_corr_labels[i]))
        im = ax[i + 8].matshow(rand_incorr_images[i], cmap = 'Grays', vmin = 0, vmax = 1)  
        ax[i+8].set_title("Label: " + str(rand_incorr_labels[i]) + "\nActual: " + str(rand_incorr_reallabels[i]))

    plt.tight_layout()

    #This line from stackoverflow https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    fig.colorbar(im, ax=ax.ravel().tolist(), label="Normalized grayscale pixel value")
    plt.suptitle("8 MNIST Images Correctly Classified (top) and Incorrectly Classified (bottom)")

# Main function
def main():
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    # Start training
    training_loss, training_accuracy, test_loss, test_accuracy, W1, W2, W3, b1, b2, b3 = train(train_loader, test_loader, epochs=100, learning_rate=0.01)
    
    plot_corr_incorr_testdata(W1, W2, W3, b1, b2, b3, test_loader)

    # PLOT TRAINING LOSS AND TEST LOSS ON ONE SUBPLOT (epoch vs loss)
    # PLOT TRAINING ACCURACY AND TEST ACCURACY ON A SECOND SUBPLOT (epoch vs accuracy)
    
    epochs_train = list(range(1, len(training_loss) + 1))  # Epochs for training loss (1, 2, ..., N)
    epochs_test = list(range(1, len(test_loss) + 1))  # Epochs for test loss (1, 2, ..., N)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Training and Test Loss on the first subplot
    ax1.plot(epochs_train, training_loss, label='Training Loss', color='blue', marker='o')
    ax1.plot(epochs_test, test_loss, label='Test Loss', color='red', marker='x')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Training and Test Accuracy on the second subplot
    ax2.plot(epochs_train, training_accuracy, label='Training Accuracy', color='blue', marker='o')
    ax2.plot(epochs_test, test_accuracy, label='Test Accuracy', color='red', marker='x')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
