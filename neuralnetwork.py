import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)


def read_and_divide_into_train_and_test(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=["Code_number"], axis=1)
    missing_to_index = df[df["Bare_Nuclei"] == "?"].index
    df.drop(missing_to_index, inplace=True)
    df.index = range(len(df))
    df["Bare_Nuclei"] = pd.to_numeric(df["Bare_Nuclei"])
    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)
    training_inputs = train.drop(columns=["Class"], axis=1)
    training_labels = train["Class"]
    test_inputs = test.drop(columns=["Class"], axis=1)
    test_labels = test["Class"]
    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    test_predictions = sigmoid(test_inputs.dot(weights))
    test_predictions[test_predictions > 0.5] = 1
    test_predictions[test_predictions <= 0.5] = 0
    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
    accuracy = tp / len(test_labels)
    return accuracy


def plot_accuracy(accuracy_array, iteration_count):
    x = range(iteration_count)
    y = accuracy_array
    plt.plot(x, y)
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    plt.close(fig=None)


def plot_loss(loss_array, iteration_count):
    x = range(iteration_count)
    y = loss_array
    plt.plot(x, y)
    plt.xlabel("# Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.close(fig=None)


def plot_corr_heatmap():
    df = pd.read_csv("breast-cancer-wisconsin.csv")
    df = df.drop(columns=["Code_number"], axis=1)
    missing_to_index = df[df["Bare_Nuclei"] == "?"].index
    df.drop(missing_to_index, inplace=True)
    df["Bare_Nuclei"] = pd.to_numeric(df["Bare_Nuclei"])
    df = df.drop(columns=["Class"], axis=1)
    plt.imshow(df.corr(), cmap="YlOrRd")
    plt.xticks(range(9), df.columns, fontsize=10, ha="right", rotation=30)
    plt.yticks(range(9), df.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=8)
    plt.title("Correlation Heatmap", fontsize=10)
    ax = plt.gca()
    for x_axis in range(9):
        for y_axis in range(9):
            ax.text(x_axis, y_axis, df.corr().values[x_axis, y_axis].round(2), ha="center",va ="center", color="b")
    plt.show()
    plt.close(fig=None)


def main():
    csv_file = "./breast-cancer-wisconsin.csv"
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        outputs = sigmoid(training_inputs.dot(weights)) #calculate outputs
        loss = training_labels - outputs.squeeze(axis=None) #calculate loss
        tuning = loss * sigmoid_derivative(outputs.squeeze(axis=None)) #calculate tuning
        train = training_inputs.T
        weights = train.dot(tuning) + weights.squeeze(axis=None) #update weights
        loss_mean = loss.mean()
        loss_array.append(loss_mean)
        accuracy = run_on_test_set(test_inputs, test_labels, weights)
        accuracy_array.append(accuracy)

    plot_accuracy(accuracy_array, iteration_count)
    plot_loss(loss_array, iteration_count)
    plot_corr_heatmap()

if __name__ == "__main__":
    main()
