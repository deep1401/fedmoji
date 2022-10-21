import os
import sys
import time
import copy
import logging
import argparse
from argparse import Namespace
import warnings
from model import LSTM
import wandb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import spacy
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from data import *
from utils import *

# args
parser = argparse.ArgumentParser("fedspeech")
parser.add_argument(
    "--data", type=str, default="special_token_data", help="location of the data corpus"
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="fedprox",
    help="specify which algorithm to use during gradient aggregation",
)
parser.add_argument(
    "--rounds", type=int, default=1000, help="number of training rounds"
)
parser.add_argument("--C", type=float, default=0.1, help="client fraction")
parser.add_argument("--K", type=int, default=100, help="number of clients")
parser.add_argument(
    "--E",
    type=int,
    default=1,
    help="number of training passes on local dataset for each round",
)
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--mu", type=float, default=0.01,
                    help="proximal term constant")
parser.add_argument(
    "--percentage",
    type=int,
    default=0,
    help="percentage of clients to have fewer than E epochs",
)
parser.add_argument(
    "--target_test_accuracy", type=float, default=99.0, help="target test accuracy"
)
parser.add_argument(
    "--balanced",
    action="store_true",
    default=False,
    help="determine if experiments must use balanced data or imbalanced one",
)
parser.add_argument(
    "--class_weights",
    action="store_true",
    default=False,
    help="determine if experiments should use class weights in loss function or not",
)
parser.add_argument(
    "--shared",
    action="store_true",
    default=False,
    help="determine if experiments must use modified causalfedgsd or not",
)
parser.add_argument(
    "--causalfedgsd",
    action="store_true",
    default=False,
    help="determine if experiments must causalfedgsd algorithm or not",
)
parser.add_argument(
    "--non_iid",
    action="store_true",
    default=False,
    help="determine if experiments should use class weights in loss function or not",
)

parser.add_argument("--save", type=str, default="exp", help="experiment name")
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="use wandb for tracking results",
)
parser.add_argument(
    "--wandb_api",
    type=str,
    default="xxxxxxxxxx",
    help="provide api key for tracking results",
)
parser.add_argument(
    "--wandb_proj_name",
    type=str,
    default="fedmoji-token-based",
    help="provide a project name",
)
parser.add_argument(
    "--wandb_run_name", type=str, default="exp", help="provide a run name"
)
parser.add_argument(
    "--wandb_run_notes", type=str, default="", help="provide notes for a run if any"
)
args = parser.parse_args()

args.save = "{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))

if args.shared and args.causalfedgsd:
    raise ValueError("Cannot use both causalfedgsd and shared")

if not os.path.exists(args.save):
    os.mkdir(args.save)
print("Experiment dir: {}".format(args.save))

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

warnings.filterwarnings("ignore")


def main():
    if not torch.cuda.is_available():
        logging.info("No GPU device available")
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    if not args.balanced:
        raw_data = {
            "train": pd.read_csv(os.path.join(args.data, "train.csv")),
            "valid": pd.read_csv(os.path.join(args.data, "dev.csv")),
            "test": pd.read_csv(os.path.join(args.data, "test.csv")),
        }
    else:
        raw_data = {
            "train": pd.read_csv(os.path.join(args.data, "balanced_train.csv")),
            "valid": pd.read_csv(os.path.join(args.data, "dev.csv")),
            "test": pd.read_csv(os.path.join(args.data, "test.csv")),
        }

    tokenizer = get_tokenizer("spacy")

    if args.shared:
        new_train, shared_data = create_global_sharing_data(
            raw_data["train"], size=0.3)
        raw_data["train"] = new_train
        raw_data["shared"] = shared_data

    data_iters = {
        "train": create_data_iter(raw_data["train"], category_dict),
        "valid": create_data_iter(raw_data["valid"], category_dict),
        "test": create_data_iter(raw_data["test"], category_dict)
    }

    vocab = build_vocab(data_iters["train"], tokenizer)

    dataset = {
        "train": create_final_dataset(data_iters["train"], vocab, tokenizer),
        "valid": create_final_dataset(data_iters["valid"], vocab, tokenizer),
        "test": create_final_dataset(data_iters["test"], vocab, tokenizer)
    }

    if args.shared:
        data_iters['shared'] = create_data_iter(
            raw_data["shared"], category_dict)

        dataset['shared'] = create_final_dataset(
            data_iters["shared"], vocab, tokenizer)

    classes = list(np.array(np.unique([elem[1] for elem in dataset["train"]])))
    classes_test = np.array(np.unique([elem[1] for elem in dataset["valid"]]))
    num_classes = len(classes)

    class_array = np.array([elem[1] for elem in dataset["train"]])
    if args.class_weights:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=classes, y=class_array
        )
    else:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=classes, y=classes
        )

    # data partition dictionary
    if args.non_iid and args.shared:
        iid_dict = non_iid_partition(dataset["train"], 100, 200, 345, 2)
    elif args.non_iid and not args.shared:
        iid_dict = non_iid_partition(dataset["train"], 100, 200, 490, 2)
    elif args.causalfedgsd and not args.shared:
        iid_dict = iid_partition_causalfedgsd(
            raw_data["train"], args.K, alpha=0.3)
    else:
        iid_dict = iid_partition(dataset["train"], args.K)

    # log the config to wandb
    config_dict = dict(
        rounds=args.rounds,
        C=args.C,
        K=args.K,
        E=args.E,
        batch_size=args.batch_size,
        lr=args.lr,
        mu=args.mu,
        percentage=args.percentage,
        target_test_accuracy=args.target_test_accuracy,
    )

    if args.wandb:
        run = wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_proj_name,
            notes=args.wandb_run_notes,
            config=config_dict,
        )

    model = LSTM(len(vocab))
    if torch.cuda.is_available():
        model.cuda()

    if args.shared:
        pre_training_dic = {
            "rounds": 20,
            "C": 1.0,
            "K": 1,
            "E": 1,
            "batch_size": args.batch_size,
            "lr": 0.01,
            "mu": 0.01,
            "percentage": 0,
            "target_test_accuracy": 99.0,
            "wandb": False,
            "algorithm": args.algorithm,
        }

        pre_training_args = Namespace(**pre_training_dic)

        shared_iid_dict = iid_partition(dataset["shared"], 1)

        pre_trained_model = training(
            pre_training_args,
            model,
            dataset["shared"],
            num_classes,
            classes,
            shared_iid_dict,
            dataset["valid"],
            class_weights,
            real_test=dataset["test"],
            pre_training=True,
        )

        emoji_trained = training(
            args,
            pre_trained_model,
            dataset["train"],
            num_classes,
            classes,
            iid_dict,
            dataset["valid"],
            class_weights,
            real_test=dataset["test"],
        )

    else:
        emoji_trained = training(
            args,
            model,
            dataset["train"],
            num_classes,
            classes,
            iid_dict,
            dataset["valid"],
            class_weights,
            real_test=dataset["test"],
        )

    if args.wandb:
        run.finish()


def GenerateLocalEpochs(percentage, size, max_epochs):
    """Generates list of epochs for selected clients to replicate system heretogeneity."""
    # if percentage is 0 then each client runs for E epochs
    if percentage == 0:
        return np.array([max_epochs] * size)
    else:
        # get the number of clients to have fewer than E epochs
        heterogenous_size = int((percentage / 100) * size)

        # generate random uniform epochs of heterogenous size between 1 and E
        epoch_list = np.random.randint(1, max_epochs, heterogenous_size)

        # the rest of the clients will have E epochs
        remaining_size = size - heterogenous_size
        rem_list = [max_epochs] * remaining_size

        epoch_list = np.append(epoch_list, rem_list, axis=0)

        # shuffle the list and return
        np.random.shuffle(epoch_list)
        logging.info("GenerateLocalEpochs done")
        return epoch_list


class ClientUpdate(object):
    def __init__(
        self,
        dataset,
        batch_size,
        learning_rate,
        epochs,
        idxs,
        mu,
        algorithm,
        class_weights,
    ):
        self.train_loader = DataLoader(
            CustomDataset(dataset, idxs),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pad_collate,
        )
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mu = mu
        self.class_weights = class_weights

    def train(self, model):
        class_weight = torch.FloatTensor(self.class_weights.tolist()).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        proximal_criterion = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, momentum=0.5
        )

        # use the weights of global model for proximal term calculation
        global_model = copy.deepcopy(model)

        # calculate local training time
        start_time = time.time()

        e_loss = []
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0

            model.train()
            for data, labels, data_len in self.train_loader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data, data_len)

                # calculate the loss + the proximal term
                _, pred = torch.max(output, 1)

                if self.algorithm == "fedprox":
                    proximal_term = 0.0

                    # iterate through the current and global model parameters
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        # update the proximal term
                        # proximal_term += torch.sum(torch.abs((w-w_t)**2))
                        proximal_term += (w - w_t).norm(2)

                    loss = criterion(output, labels) + \
                        (self.mu / 2) * proximal_term
                else:
                    loss = criterion(output, labels)

                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)

            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

        total_loss = sum(e_loss) / len(e_loss)

        return model.state_dict(), total_loss, (time.time() - start_time)


def training(
    args,
    model,
    dataset,
    num_classes,
    classes,
    data_dict,
    test_data_dict,
    class_weights,
    real_test=None,
    pre_training=False,
):
    """Implements Federated Averaging Algorithm from the FedAvg paper."""
    # global model weights
    global_weights = model.state_dict()

    # training loss
    train_loss = []

    # test accuracy
    test_acc = []

    # store last loss for convergence
    last_loss = 0.0

    # Set to Pre-training
    if pre_training:
        print("Pre-training Model first")
    # assign test set
    if real_test is None:
        logging.info("Test Set not specified")
        real_test = test_data_dict

    # total time taken
    total_time = 0

    logging.info(f"System heterogeneity set to {args.percentage}% stragglers.")
    logging.info(
        f"Picking {max(int(args.C * args.K), 1)} random clients per round.")

    for curr_round in range(1, args.rounds + 1):
        w, local_loss, lst_local_train_time = [], [], []
        m = max(int(args.C * args.K), 1)

        heterogenous_epoch_list = GenerateLocalEpochs(
            args.percentage, size=m, max_epochs=args.E
        )
        heterogenous_epoch_list = np.array(heterogenous_epoch_list)

        S_t = np.random.choice(range(args.K), m, replace=False)
        S_t = np.array(S_t)

        # drop all the clients that are stragglers in case of federated averaging
        if args.algorithm == "fedavg":
            stragglers_indices = np.argwhere(heterogenous_epoch_list < args.E)
            heterogenous_epoch_list = np.delete(
                heterogenous_epoch_list, stragglers_indices
            )
            S_t = np.delete(S_t, stragglers_indices)

        for k, epoch in zip(S_t, heterogenous_epoch_list):
            local_update = ClientUpdate(
                dataset=dataset,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                epochs=epoch,
                idxs=data_dict[k],
                mu=args.mu,
                algorithm=args.algorithm,
                class_weights=class_weights,
            )
            weights, loss, local_train_time = local_update.train(
                model=copy.deepcopy(model)
            )

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
            lst_local_train_time.append(local_train_time)

        # calculate time to update the global weights
        global_start_time = time.time()

        # updating the global weights
        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]
            weights_avg[k] = torch.div(weights_avg[k], len(w))
        global_weights = weights_avg

        global_end_time = time.time()

        # calculate total time
        total_time += (global_end_time - global_start_time) + sum(
            lst_local_train_time
        ) / len(lst_local_train_time)

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)

        # test accuracy
        criterion = nn.CrossEntropyLoss()
        (
            test_loss,
            test_accuracy,
            auc,
            precision_full,
            recall_full,
            f1_score_full,
            mcc,
        ) = testing(
            model, test_data_dict, args.batch_size, criterion, num_classes, classes
        )
        (
            real_test_loss,
            real_test_accuracy,
            real_auc,
            real_precision_full,
            real_recall_full,
            real_f1_score_full,
            real_mcc,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        logging.info(
            f"Round: {curr_round}... \tAverage Train Loss: {round(loss_avg, 3)}... \tTest Loss: {test_loss}... \tTest Accuracy: {test_accuracy}... \tAUC Score: {auc}... \tPrecision: {precision_full}... \tRecall: {recall_full}... \tF1: {f1_score_full}... \tMCC: {mcc}"
        )
        train_loss.append(loss_avg)
        test_acc.append(test_accuracy)

        # break if we achieve the target test accuracy
        if test_accuracy >= args.target_test_accuracy:
            (
                real_test_loss,
                real_test_accuracy,
                real_auc,
                real_precision_full,
                real_recall_full,
                real_f1_score_full,
                real_mcc,
            ) = testing(
                model, real_test, args.batch_size, criterion, num_classes, classes
            )
            logging.info(
                f"FINAL TESTING\n... \tTest Loss: {real_test_loss}... \tTest Accuracy: {real_test_accuracy}... \tAUC Score: {real_auc}... \tPrecision: {real_precision_full}... \tRecall: {real_recall_full}... \tF1: {real_f1_score_full}... \tMCC: {real_mcc}"
            )
            if args.wandb and not pre_training:
                wb_metrics = {
                    "round": curr_round,
                    "avg_train_loss": loss_avg,
                    "valid/loss": test_loss,
                    "valid/accuracy": test_accuracy,
                    "valid/auc_score": auc,
                    "valid/f1": f1_score_full,
                    "valid/mcc": mcc,
                    "valid/precision": precision_full,
                    "valid/recall": recall_full,
                    "test/loss": real_test_loss,
                    "test/accuracy": real_test_accuracy,
                    "test/auc_score": real_auc,
                    "test/f1": real_f1_score_full,
                    "test/mcc": real_mcc,
                    "test/precision": real_precision_full,
                    "test/recall": real_recall_full,
                }
                wandb.log(wb_metrics, step=curr_round)
            rounds = curr_round
            break

        # final Test
        if curr_round == args.rounds:
            (
                real_test_loss,
                real_test_accuracy,
                real_auc,
                real_precision_full,
                real_recall_full,
                real_f1_score_full,
                real_mcc,
            ) = testing(
                model, real_test, args.batch_size, criterion, num_classes, classes
            )
            logging.info(
                f"FINAL TESTING\n... \tTest Loss: {real_test_loss}... \tTest Accuracy: {real_test_accuracy}... \tAUC Score: {real_auc}... \tPrecision: {real_precision_full}... \tRecall: {real_recall_full}... \tF1: {real_f1_score_full}... \tMCC: {real_mcc}"
            )
        if args.wandb and not pre_training:
            wb_metrics = {
                "round": curr_round,
                "avg_train_loss": loss_avg,
                "valid/loss": test_loss,
                "valid/accuracy": test_accuracy,
                "valid/auc_score": auc,
                "valid/f1": f1_score_full,
                "valid/mcc": mcc,
                "valid/precision": precision_full,
                "valid/recall": recall_full,
                "test/loss": real_test_loss,
                "test/accuracy": real_test_accuracy,
                "test/auc_score": real_auc,
                "test/f1": real_f1_score_full,
                "test/mcc": real_mcc,
                "test/precision": real_precision_full,
                "test/recall": real_recall_full,
            }
            wandb.log(wb_metrics, step=curr_round)

        # update the last loss
        last_loss = loss_avg
    if pre_training:
        print("Pre-Training Complete")
    return model


def testing(model, dataset, batch_size, criterion, num_classes, classes):
    # test loss
    test_loss = 0.0
    auc, precision_full, recall_full, f1_score_full, mcc = 0.0, 0.0, 0.0, 0.0, 0.0

    correct_class = list(0.0 for i in range(num_classes))
    total_class = list(0.0 for i in range(num_classes))

    test_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate)

    model.eval()
    for data, labels, data_len in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data, data_len)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)
        auc_temp = roc_auc_score(
            labels.cpu(),
            torch.exp(output).cpu().detach().numpy(),
            multi_class="ovo",
            labels=classes,
        )
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            labels.cpu(),
            torch.argmax(torch.exp(output), 1).cpu().detach().numpy(),
            average="weighted",
            labels=classes,
        )
        mcc_temp = matthews_corrcoef(
            labels.cpu(), torch.argmax(torch.exp(output), 1).cpu().detach().numpy()
        )
        if auc_temp < 1.0:
            auc += auc_temp
        if precision is not None and precision <= 1.0:
            precision_full += precision
        else:
            logging.info(f"Precision Anomaly: {precision}")
        if recall is not None and recall <= 1.0:
            recall_full += recall
        else:
            logging.info(f"Recall Anomaly: {recall}")
        if f1_score is not None and f1_score <= 1.0:
            f1_score_full += f1_score
        else:
            logging.info(f"F1 Score Anomaly: {f1_score}")
        if mcc_temp is not None and mcc_temp <= 1.0:
            mcc += mcc_temp
        else:
            logging.info(f"MCC Anomaly: {mcc_temp}")

        _, pred = torch.max(output, 1)

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = (
            np.squeeze(correct_tensor.numpy())
            if not torch.cuda.is_available()
            else np.squeeze(correct_tensor.cpu().numpy())
        )

        # test accuracy for each object class
        for i in range(num_classes):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1

    num_batches = int(len(test_loader.dataset) / batch_size)
    test_loss = test_loss / len(test_loader.dataset)
    auc = auc / num_batches
    precision_full = precision_full / num_batches
    recall_full = recall_full / num_batches
    f1_score_full = f1_score_full / num_batches
    mcc = mcc / num_batches
    test_accuracy = 100.0 * np.sum(correct_class) / np.sum(total_class)

    return (
        test_loss,
        test_accuracy,
        auc,
        precision_full,
        recall_full,
        f1_score_full,
        mcc,
    )


if __name__ == "__main__":
    main()
