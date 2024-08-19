import utils as ut
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch_geometric
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm
from sgnn.model_sgnn import SiameseNetwork, ContrastiveLoss

num_features = 333  # Num of Rois
num_pairs = 40  # Num of pairs by observation
threshold = 0.5
NUM_EPOCHS = 40


#df.iloc[:, :-1] = np.tanh(df.iloc[:, :-1])

def Siamese_test(test_data, threshold, num_pairs):  # threshold = margin
    total_loss = 0
    pred = []
    label = []
    scores = []
    for y in test_data:
        model.eval()
        output1, output2 = model(y[0], y[1])
        loss = criterion(output1, output2, y[0].y)  # Does not matter each y[0].y or y[1].y
        total_loss += loss.item()

        pred.append(np.where(nn.functional.pairwise_distance(output1, output2).detach().numpy() > threshold, 1, 0))
        scores.append(nn.functional.pairwise_distance(output1, output2).detach().numpy())
        label.append(y[0].y.detach().numpy())  # Does not matter each y[0].y or y[1].y

    y_pred = np.array(pred).ravel()
    y_true = np.array(label).ravel()
    y_scores = np.array(scores).ravel()

    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    epoch_rec = tp / (tp + fn)
    epoch_prec = tp / (tp + fp)
    epoch_f1 = 2 * (epoch_rec * epoch_prec) / (epoch_rec + epoch_prec)
    epoch_spe = tn / (tn + fp)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)
    acc_balanced = roc_auc_score(y_true, y_scores)

    print('y_pred', len(y_pred[y_pred == 1]) / len(y_pred))
    print('y_true', len(y_true[y_true == 1]) / len(y_true))

    return epoch_rec, epoch_prec, epoch_acc, total_loss / len(label), acc_balanced, epoch_f1, y_pred, y_true, y_scores


#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(device)
for time in ['after', 'before']:
    metrics_dict = {"fold": [], "auc": [], "accuracy": [], "f1_score": [], "precision": [], "recall": []}

    df = pd.read_csv('/Users/rodrigo/Documents/data/ayahuasca_acute_preacute/corr_matrices.csv').drop(columns='Unnamed: 0')
    HRS = pd.read_csv('/Users/rodrigo/Documents/data/ayahuasca_acute_preacute/Ayahuasca_HRS.csv')
    condition = (
    ((df['Group'] == 'J') & (df['Subject'].isin([7, 9, 18]))) |     # Examinate subject 18
    ((df['Group'] == 'O') & (df['Subject'].isin([1, 7])))
    )

    # Inverting the condition to keep rows that do NOT match the condition
    df = df[~condition]

    df['Group'] = df['Group'].astype(str).replace({'O': 0, 'J': 1})

    df = df[df['Time'] == time]

    df = df.drop(columns='Time')

    print('time',time)

    for fold in range(1, 6):
        print('----------------------------------------------------------------------')
        print('Fold {}'.format(fold))

        np.random.seed(fold)
        arr = np.random.choice(df['Subject'].unique(), 10)

        df_train = df[~df.Subject.isin(arr)].drop(columns='Subject')
        df_test = df[df.Subject.isin(arr)].drop(columns='Subject')

        pair1, pair2, labels, pair1_test, pair2_test, labels_test = ut.create_pairs(df_train, df_test,
                                                                                    n_pair_per_observation=num_pairs)

        A = ut.reconstruct_symmetric_matrix(num_features, df_train.iloc[:, :-1].mean(axis=0))

        train_data_1, val_data_1 = ut.create_graph(pd.DataFrame(pair1),
                                                pd.DataFrame(pair1), pd.DataFrame(labels), pd.DataFrame(labels),
                                                size=num_features,
                                                method={'knn_group': ut.compute_KNN_graph(A,
                                                                                            10)})

        del pair1

        train_data_2, val_data_2 = ut.create_graph(pd.DataFrame(pair2),
                                                pd.DataFrame(pair2), pd.DataFrame(labels), pd.DataFrame(labels),
                                                size=num_features,
                                                method={'knn_group': ut.compute_KNN_graph(A,
                                                                                            10)})
        del pair2

        test_data_1, test_data_1 = ut.create_graph(pd.DataFrame(pair1_test), pd.DataFrame(pair1_test),
                                                pd.DataFrame(labels_test), pd.DataFrame(labels_test), size=num_features,
                                                method={'knn_group': ut.compute_KNN_graph(A,
                                                                                            10)})

        del pair1_test

        test_data_2, test_data_2 = ut.create_graph(pd.DataFrame(pair2_test),
                                                pd.DataFrame(pair2_test), pd.DataFrame(labels_test),
                                                pd.DataFrame(labels_test), size=num_features,
                                                method={'knn_group': ut.compute_KNN_graph(A,
                                                                                            10)})

        del pair2_test

        train_loader_1, val_loader_1 = ut.create_batch(train_data_1, val_data_1, batch_size=32)
        train_loader_2, val_loader_2 = ut.create_batch(train_data_2, val_data_2, batch_size=32)
        test_loader_1, test_loader_1 = ut.create_batch(test_data_1, test_data_1, batch_size=32)
        test_loader_2, test_loader_2 = ut.create_batch(test_data_2, test_data_2, batch_size=32)

        del test_data_1, test_data_2, train_data_1, train_data_2, val_loader_1, val_loader_2, val_data_1, val_data_2

        print('Test size', len(labels_test))
        print('Train size', len(labels))

        print('Ratio of similars in test {}'.format((np.array(labels_test) == 0.0).sum() / len(labels_test)))
        print('Ratio of similars in train {}'.format((np.array(labels) == 0.0).sum() / len(labels)))

        data = zip(train_loader_1, train_loader_2)
        test_data = zip(test_loader_1, test_loader_2)

        model = SiameseNetwork(333, k_order=3, dropout=0.7).to(device)
        criterion = ContrastiveLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=True)

        metrics = {"loss_train": [], "loss_test": [], "roc_auc": [], "acc_train": []}

        def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Num Parameters:', count_parameters(model))

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            loop = tqdm(zip(train_loader_1, train_loader_2))  # data
            total_loss = 0.0

            for (x, y) in enumerate(loop):
                optimizer.zero_grad()
                output1, output2 = model(y[0].to(device), y[1].to(device))
                loss = criterion(output1, output2, y[0].y)  # Does not matter each y[0].y or y[1].y
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
                loop.set_postfix(loss=total_loss / len(labels))

                #scheduler.step()

            test_data = zip(test_loader_1, test_loader_2)
            test_rec, test_prec, test_acc, test_loss, roc_auc, test_f1, y_pred, y_true, y_scores = Siamese_test(test_data, threshold, num_pairs)
            
            metrics['loss_test'].append(test_loss)
            metrics['roc_auc'].append(roc_auc)
            
            print('Val ROC_AUC {} , Val Loss {}'.format(roc_auc, test_loss))
            print('Val Recall {} , Val Loss {}'.format(test_rec, test_loss))
            print('Val Precision {} , Val Loss {}'.format(test_prec, test_loss))
            print('Val acc {} , Val f1 {}'.format(test_acc, test_f1))

        test_data = zip(test_loader_1, test_loader_2)
        test_rec, test_prec, test_acc, test_loss, roc_auc, test_f1, y_pred, y_true, y_scores = Siamese_test(test_data, threshold=.3, num_pairs=num_pairs)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)

        metrics_dict["fold"].append(fold)
        metrics_dict["auc"].append(roc_auc)
        metrics_dict["accuracy"].append(test_acc)
        metrics_dict["f1_score"].append(test_f1)
        metrics_dict["precision"].append(test_prec)
        metrics_dict["recall"].append(test_rec)

        print(f'Fold {fold} metrics: AUC: {roc_auc}, Accuracy: {test_acc}, F1 Score: {test_f1}, Precision: {test_prec}, Recall: {test_rec}')

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv('fold_metrics_{}.csv'.format(time), index=False)
    print("Metrics saved to fold_metrics.csv")