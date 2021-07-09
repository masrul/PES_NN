import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PESData:
    def __init__(self, distFile, energyFile):
        self.distFile = distFile
        self.energyFile = energyFile

    def removeOutLier(self):
        print("Total data: %d" % len(self.energies))
        mean = np.mean(self.energies)
        std = np.std(self.energies)

        low = mean - std
        high = mean + std

        temp_enr = []
        temp_distance = []

        for i in range(len(self.energies)):

            if self.energies[i] < high and self.energies[i] > low:
                temp_enr.append(self.energies[i])
                temp_distance.append(self.distances[i, :])

        self.energies = np.array(temp_enr)
        self.distances = np.array(temp_distance)
        print("Total data: %d" % len(self.energies))

    def read(self):
        self.distances = np.genfromtxt(self.distFile)
        self.energies = np.genfromtxt(self.energyFile)

        self.energies -= np.min(self.energies)
        self.nsamples, self.nfeatures = self.distances.shape
        self.energies = self.energies.reshape(self.nsamples, 1)

        self.dist_scaler = MinMaxScaler()
        self.dist_scaler.fit(self.distances)
        self.distances = self.dist_scaler.transform(self.distances)

        self.enrg_scaler = MinMaxScaler()
        self.enrg_scaler.fit(self.energies.reshape(self.nsamples, 1))
        self.energies = self.enrg_scaler.transform(
            self.energies.reshape(self.nsamples, 1)
        )

        (
            self.distance_train,
            self.distance_test,
            self.energies_train,
            self.energies_test,
        ) = train_test_split(
            self.distances,
            self.energies,
            test_size=0.50,
            shuffle=True,
            random_state=1234,
        )

        self.nsamples_train = len(self.energies_train)
        self.nsamples_test = len(self.energies_test)

        print(self.distance_train.shape)
        print(self.energies_train.shape)
        print(self.distance_test.shape)
        print(self.energies_test.shape)

    @property
    def X_train(self):
        return torch.tensor(self.distance_train, dtype=torch.float32).reshape(
            self.nsamples_train, self.nfeatures
        )

    @property
    def y_train(self):
        return torch.tensor(self.energies_train, dtype=torch.float32).reshape(
            self.nsamples_train, 1
        )

    @property
    def X_test(self):
        return torch.tensor(self.distance_test, dtype=torch.float32)

    @property
    def y_test(self):
        return torch.tensor(self.energies_test, dtype=torch.float32).reshape(
            self.nsamples_train, 1
        )


class NeuralNet(nn.Module):
    def __init__(self, layer_arch):

        super(NeuralNet, self).__init__()

        nlayers = len(layer_arch)
        self.layers = [None] * (nlayers - 1)
        self.activations = [None] * (nlayers - 2)

        torch.manual_seed(786)
        for i in range(nlayers - 1):
            self.layers[i] = nn.Linear(layer_arch[i], layer_arch[i + 1])
            torch.nn.init.zeros_(self.layers[i].bias)
            torch.nn.init.xavier_uniform_(self.layers[i].weight)

        for i in range(nlayers - 2):
            self.activations[i] = nn.ReLU()

        self.layers = nn.ModuleList(self.layers)
        self.activations = nn.ModuleList(self.activations)

    def forward(self, x):

        out = x
        for layer, relu in zip(self.layers[:-1], self.activations):
            out = layer(out)
            out = relu(out)

        layer = self.layers[-1]
        out = layer(out)

        return out


@torch.no_grad()
def actual_predict(my_data, net, img_num):
    fig, axs = plt.subplots(1, 2)

    # train
    dist_train = my_data.X_train

    energy_predicts_train = net(dist_train.to(device))
    energy_predicts_train = my_data.enrg_scaler.inverse_transform(energy_predicts_train)

    energy_actual_train = my_data.enrg_scaler.inverse_transform(my_data.energies_train)

    # test
    dist_test = my_data.X_test

    energy_predicts_test = net(dist_test.to(device))
    energy_predicts_test = my_data.enrg_scaler.inverse_transform(energy_predicts_test)

    energy_actual_test = my_data.enrg_scaler.inverse_transform(my_data.energies_test)

    # train
    axs[0].scatter(energy_actual_train, energy_predicts_train, color="red")
    axs[1].scatter(energy_actual_test, energy_predicts_test, color="blue")

    maxval = max(energy_actual_train)
    minval = min(energy_actual_train)
    a = np.linspace(minval, maxval)

    axs[0].plot(a, a, color="black", linewidth=1.5)
    axs[1].plot(a, a, color="black", linewidth=1.5)
    axs[0].set_aspect("equal", "box")
    axs[1].set_aspect("equal", "box")
    axs[0].set_xlim(minval, maxval)
    axs[0].set_ylim(minval, maxval)
    axs[1].set_xlim(minval, maxval)
    axs[1].set_ylim(minval, maxval)

    plt.savefig(
        "images/img" + str(img_num).zfill(5) + ".png", bbox_inches="tight", dpi=300,
    )
    plt.close()


def main():
    my_data = PESData("dist_eoh.dat", "energy_eoh.dat")
    my_data.read()

    nepochs = 10000

    net = NeuralNet(layer_arch=(my_data.nfeatures, 200, 200, 1))
    criterion = torch.nn.MSELoss(reduction="sum")
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    nbatchs = 10
    batch_size = int(my_data.nsamples_train / nbatchs)
    img_num = 0

    for epoch in range(nepochs):

        # forward pass
        for ibatch in range(nbatchs):
            sIDx = ibatch * batch_size
            eIDx = (ibatch + 1) * batch_size

            batch_train_X = my_data.X_train[sIDx:eIDx, :]
            batch_train_y = my_data.y_train[sIDx:eIDx, :]

            outputs = net(batch_train_X.to(device))
            loss = criterion(outputs, batch_train_y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{nepochs}], Loss: {loss.item():.4f}")

        if (epoch + 1) % 10 == 0 or epoch < 20:
            img_num += 1
            actual_predict(my_data, net, img_num)


main()
