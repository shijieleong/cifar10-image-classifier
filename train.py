import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import load_data
from neural_net import NeuralNet
import utils


def train(net, train_loader, device):
    learning_rate = 0.001
    momentum=0.9
    n_epoch=30

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Start training
    for epoch in range(n_epoch):
        print(f"Training epoch {epoch}...")
        
        # Reset running loss
        running_loss = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Feed Forward Neural Network
            output = net(images)

            # Calculate loss
            loss = loss_function(output, labels)
            loss.backward() # Backpropagation
            optimizer.step() # Increment step by learning rate

            running_loss += loss.item()

        # Finished training of one batch
        print(f'Loss: {running_loss/len(train_loader):.4f}')


def evaluate(net, test_loader, device):
    # Set the network to evaluation mode
    net.eval()

    # Start evaluating
    total = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # Make predictions with model
            output = net(images)
            _, predicted = torch.max(output, 1)

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')


def main():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    train_loader, test_loader = load_data()

    # Create neural network, loss function and optimizer
    net = NeuralNet()
    #net = utils.load_checkpoint(net, './models/trained_model.pth')
    train(net, train_loader, device)
    
    # Save model
    utils.save_checkpoint(net)

    # Evaluate accuracy
    evaluate(net, test_loader, device)

    return


if __name__ == "__main__":
    main()



