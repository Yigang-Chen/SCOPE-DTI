import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm
import time
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import base64
import json
from utils import print_with_time
# new
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.writer = SummaryWriter(log_dir=self.output_dir)
        # Learning rate scheduler 
        #self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=10, min_lr=1e-6, verbose=True)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1

            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            # Log training loss to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)

            auroc, auprc, val_loss = self.test(dataloader="val")
            
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            # Log validation metrics to TensorBoard
            self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
            self.writer.add_scalar('AUROC/val', auroc, self.current_epoch)
            self.writer.add_scalar('AUPRC/val', auprc, self.current_epoch)

            # Step the scheduler based on validation loss 
            #self.scheduler.step(val_loss)

            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print_with_time('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))

            # if len(self.val_loss_epoch) > 10 and all(
            #         val >= min(self.val_loss_epoch[:-10]) for val in self.val_loss_epoch[-10:]):
            #     print("Early stopping triggered.")
            #     break

        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        print_with_time('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()

        self.writer.close()

        return self.test_metrics

    def save_result(self):
        def convert_to_json_friendly(data):
            if isinstance(data, np.generic):  # For NumPy scalar types
                return data.item()  # Convert to standard Python type (float, int)
            elif isinstance(data, bytes):  # For binary data
                return base64.b64encode(data).decode('utf-8')  # Encode to base64 string
            elif isinstance(data, dict):  # If it's a dictionary
                return {k: convert_to_json_friendly(v) for k, v in data.items()}
            elif isinstance(data, list):  # If it's a list
                return [convert_to_json_friendly(item) for item in data]
            else:
                return data  # Return the data as is if it's a standard type
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

        self.test_metrics = convert_to_json_friendly(self.test_metrics)
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as wf:
            json.dump(self.test_metrics, wf, indent=4)

    def train_epoch(self):


        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)

        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):

            self.step += 1
            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)

            self.optim.zero_grad()
            v_d_out, v_p_out, f, score = self.model(v_d, v_p)
            
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()

            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 加入梯度裁剪，防止梯度爆炸
            
            self.optim.step()
            loss_epoch += loss.item()


        loss_epoch = loss_epoch / num_batches
        print_with_time('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss