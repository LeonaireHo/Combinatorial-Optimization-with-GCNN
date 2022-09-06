import os
from GCN import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import init
from tabulate import tabulate
import csv
from focal_Loss import *
from utility import *
from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def valide(model, criterion, data_validation, proportion=0.7):
    vloss = []
    accuracy_value = []
    accuracy_instance = []

    accuracy_one = []
    accuracy_one_predict = []
    nb_predict_one = []
    nb_one = []

    accuracy_zero = []
    accuracy_zero_predict = []
    nb_zero = []
    nb_predict_zero = []

    accuracy_fixe_one = []
    accuracy_fixe_zero = []
    accuracy_fixe = []

    data_size = len(data_validation)
    for i in range(data_size):
        if device == torch.device("cuda"):
            inputs, labels = data_validation[i][0], data_validation[i][
                1]  # torch.from_numpy(data_train[i]['X']).float(),torch.from_numpy(data_train[i]['Y']).type(torch.FloatTensor).float()
            A = data_validation[i][2]  # torch.from_numpy(data_train[i]['A']).float()
        else:
            inputs, labels = torch.from_numpy(data_validation[i]['X']).float(), torch.from_numpy(
                data_validation[i]['Y']).type(torch.FloatTensor).float()
            A = torch.from_numpy(data_validation[i]['A']).float()
        predictions = model(inputs, A)[:len(labels)].squeeze(dim=-1)
        loss = criterion(predictions, labels)
        A = A.cpu()
        labels = labels.cpu()
        predictions = predictions.cpu()

        nb_one_total = torch.sum(torch.where(labels == 1, 1, 0))
        nb_zero_total = (len(labels) - nb_one_total)
        aux = torch.Tensor([0.5])
        y_hat = (predictions > aux).float() * 1
        is_equal = torch.where(y_hat == labels, 1, 0)
        vloss.append(loss.cpu().detach().numpy())
        accuracy_value.append(is_equal.cpu().sum().detach().numpy() /
                              len(labels))
        accuracy_instance.append(int(accuracy_value[-1] == 1))
        # accuracy proportion
        predictions_abs = np.abs(0.5 - predictions.detach().numpy())
        thresholds = abs(np.sort(-predictions_abs))[int(proportion * len(predictions.cpu().detach().numpy())) - 1]
        #             thresholds_low = np.sort(predictions_abs)[int(proportion_low * nb_zero_total]
        #             thresholds_up = abs(np.sort(-predictions_abs))[int(proportion_up * nb_one_total)]

        # accuracy for differents labels(0 or 1)
        nb_one_fixe = 0
        nb_zero_fixe = 0
        nb_correct_one = 0
        nb_correct_zero = 0
        nb_correct_fixe_one = 0
        nb_correct_fixe_zero = 0
        nb_correct_fixe = 0
        nb_fixe = 0
        for j, v in enumerate(is_equal):
            if labels[j] == 1:
                nb_correct_one += v
            else:
                nb_correct_zero += v

            if predictions_abs[j] >= thresholds:
                if labels[j]:
                    nb_correct_fixe_one += v
                else:
                    nb_correct_fixe_zero += v

                if y_hat[j]:
                    nb_one_fixe += 1
                else:
                    nb_zero_fixe += 1

        nb_one_total = torch.sum(torch.where(labels == 1, 1, 0))
        nb_one_predict = torch.sum(torch.where(y_hat == 1, 1, 0))
        accuracy_one.append((nb_correct_one / nb_one_total))
        accuracy_one_predict.append((nb_correct_one / nb_one_predict) if nb_one_predict > 0 else 0)
        nb_predict_one.append(nb_one_predict)
        nb_one.append(nb_one_total)

        nb_zero_predict = (len(labels) - nb_one_predict)
        accuracy_zero.append((nb_correct_zero / nb_zero_total))
        accuracy_zero_predict.append((nb_correct_zero / nb_zero_predict) if nb_zero_predict > 0 else 0)
        nb_predict_zero.append(nb_zero_predict)
        nb_zero.append(nb_zero_total)

        accuracy_fixe.append((nb_correct_fixe_one + nb_correct_fixe_zero) / (nb_one_fixe + nb_zero_fixe))
        accuracy_fixe_one.append(nb_correct_fixe_one / nb_one_fixe if nb_one_fixe > 0 else 0)
        accuracy_fixe_zero.append(nb_correct_fixe_zero / nb_zero_fixe if nb_zero_fixe > 0 else 0)

    #     print(nb_correct_fixe.cpu(),nb_var_fixe)
    #         print(labels,"\n",predictions)
    #         print(labels[0:10],y_hat[0:10])
    msg = ("Loss = %f,Accuracy_value = %f,Accuracy_instance = %f\n" \
           % (np.mean(vloss), np.mean(accuracy_value), np.sum(accuracy_instance) / len(accuracy_instance)))
    msg += ("Label_One_Predict_One = %f,Predict_One_Label_One = %f,Nombre_Predict_One = %f,Nombre_One_Mean = %f\n" \
            % (np.mean(accuracy_one), np.mean(accuracy_one_predict), np.mean(nb_predict_one), np.mean(nb_one)))
    msg += (
                "Label_Zero_Predict_Zero = %f,Label_Zero_Predict_Zero = %f,Nombre_Predict_Zero = %f,Nombre_Zero_Mean = %f\n" \
                % (
                np.mean(accuracy_zero), np.mean(accuracy_zero_predict), np.mean(nb_predict_zero), np.mean(nb_zero)))
    msg += ("Accuracy_fixe_mean = %f,Accuracy_fixe_one = %f,Accuracy_fixe_zero = %f\n" \
            % (np.mean(accuracy_fixe), np.mean(accuracy_fixe_one), np.mean(accuracy_fixe_zero)))
    print(msg)
    return msg, np.mean(accuracy_value), np.sum(accuracy_instance) / len(accuracy_instance), np.mean(
        accuracy_fixe_one), np.mean(accuracy_fixe_zero)


def train(net, criterion, optm,data_train, data_valide, proportion, writer, scheduler = None,batch_size=1, EPOCHS=500, do_valide=True, do_log=True,
          epochs_continue=0):
    aux = torch.Tensor([0.5]).to(device)
    for epoch in tqdm(range(epochs_continue, epochs_continue + EPOCHS)):
        # validation
        if do_valide and epoch % 50 == 0:
            v = valide(net, criterion, data_valide, proportion)
            if do_log:
                writer.add_scalar('Accuracy_fixe_one ', v[3], epoch)
                writer.add_scalar('Accuracy_fixe_zero ', v[4], epoch)
        # train
        data_size = len(data_train)
        log = []
        for i in range(data_size):
            if device == torch.device("cuda"):
                train_inputs, train_labels = data_train[i][0], data_train[i][1]
                # torch.from_numpy(data_train[i]['X']).float(),torch.from_numpy(data_train[i]['Y']).type(torch.FloatTensor).float()
                A = data_train[i][2]  # torch.from_numpy(data_train[i]['A']).float()
            else:
                train_inputs, train_labels = torch.from_numpy(data_train[i]['X']).float(), torch.from_numpy(
                    data_train[i]['Y']).type(torch.FloatTensor).float()
                A = torch.from_numpy(data_train[i]['A']).float()

            predictions = net(train_inputs, A)[:len(train_labels)].squeeze(dim=-1)
            loss = criterion(predictions, train_labels)
            loss.backward()
            if epoch % batch_size == 0:
                optm.step()
                optm.zero_grad()
            writer.add_scalar('LearningRate', optm.param_groups[0]['lr'], epoch)
            if scheduler is not None:
                scheduler.step()
            gradient_norm = 0
            for p in net.parameters():
                param_norm = p.grad.detach().data.norm(2)
                gradient_norm += param_norm.item() ** 2
            gradient_norm = gradient_norm ** 0.5

            y_hat = (predictions > aux).float() * 1
            is_equal = torch.where(y_hat == train_labels, 1, 0)
            acc = is_equal.cpu().sum().detach().numpy() / len(train_labels)
            log.append([loss.cpu().detach().numpy(), acc, gradient_norm])

        log = np.array(log).T
        if do_log:
            writer.add_scalar('Loss', np.mean(log[0]), epoch)
            writer.add_scalar('Accuracy', np.mean(log[1]), epoch)
            writer.add_scalar('Gradient_norm', np.mean(log[2]), epoch)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # data_name = "DataSetSC1000"
    data_name = "DataSetCFL30"
    data_path = data_name+"/"
    print("Loading data "+data_name)
    dataSet = load_data(data_path)
    split = [0.70,0.15,0.15]
    data_train, data_validation, data_test = torch.utils.data.random_split(dataSet,
                                                                           [int(len(dataSet) * s) for s in split])
    #upload data to cuda if is available
    if device == torch.device("cuda"):
        print("Upload data to cuda")
        data_train_gpu = []
        for i in range(len(data_train)):
            inputs, labels = torch.from_numpy(data_train[i]['X']).float(), torch.from_numpy(data_train[i]['Y']).type(
                torch.FloatTensor).float()
            A = torch.from_numpy(data_train[i]['A']).float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            A = A.to(device)
            data_train_gpu.append([inputs, labels, A])

        data_valide_gpu = []
        for i in range(len(data_validation)):
            inputs, labels = torch.from_numpy(data_validation[i]['X']).float(), torch.from_numpy(
                data_validation[i]['Y']).type(torch.FloatTensor).float()
            A = torch.from_numpy(data_validation[i]['A']).float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            A = A.to(device)
            data_valide_gpu.append([inputs, labels, A])

        data_test_gpu = []
        for i in range(len(data_test)):
            inputs, labels = torch.from_numpy(data_test[i]['X']).float(), torch.from_numpy(data_test[i]['Y']).type(
                torch.FloatTensor).float()
            A = torch.from_numpy(data_test[i]['A']).float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            A = A.to(device)
            data_test_gpu.append([inputs, labels, A])

    nb_mlp = 3
    proportion = 0.8
    feature_size = 24
    H = 50
    info_valide = []
    gamma = 0.5
    alpha = 0.25
    batch_size = 1
    EPOCHS = 100
    learning_rate = 0.001
    criterion = nn.BCELoss()
    log_path = './logs/CFL/'
    #     criterion = FocalLoss(gamma=gamma,alpha = alpha)
    for nb_mlp in [6]:
        if criterion.__str__() == "BCELoss()":
            train_name = data_name + "_BCE_" + \
                         nb_mlp.__str__() + "MLP_" + H.__str__() + "TailleH_" + batch_size.__str__() + "batch_size_withscheduler" + "layernorm+cat"
        elif criterion.__str__() == "FocalLoss()":
            train_name = data_name + "_FL_" + gamma.__str__() + "Gamma_" + alpha.__str__() + "Alpha_" + \
                         nb_mlp.__str__() + "MLP_" + H.__str__() + "TailleH_" + batch_size.__str__() + "batch_size_withscheduler" + "layernorm+cat"
        log_file = log_path+ train_name + "/"
        writer = SummaryWriter(log_file)
        net = VariablePredictor(feature_size, H, nb_mlp)
        net.to(device)
        optm = Adam(net.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optm, gamma=1 - 10e-6)
        train(net, criterion, optm,data_train_gpu,data_valide_gpu, proportion, writer, batch_size=batch_size, EPOCHS=EPOCHS,
              do_valide=True, do_log=True)
        model_path = "model/" + train_name
        torch.save(net.state_dict(), model_path)
        msg = valide(net, criterion, data_test_gpu, proportion=proportion)
        info_valide.append(msg[1:])
        with open(log_file+ "/valide_log.txt", "w") as txtfile:
            print("{}".format(msg), file=txtfile)

