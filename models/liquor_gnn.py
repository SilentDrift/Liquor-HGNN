import datetime
import os
import random
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wntr
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero, SAGEConv, GraphConv, LEConv
from torch_geometric.nn.dense.linear import Linear

from dataset import get_link


def generate_date_prefix(random_letters=True) -> str:
    out = f'{str(datetime.date.today())}_{datetime.datetime.now().hour}-{datetime.datetime.now().minute}'
    if random_letters:
        t = 1000 * time.time()
        random.seed(int(t) % 2 ** 32)
        out = f'{out}_{"".join(random.choices(string.ascii_uppercase, k=5))}'
    return out


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, args):
        super().__init__()
        self.args = args
        if args.layer == "GATConv":
            layer = GATConv
            kwargs = {"in_channels": (-1, -1), "out_channels": hidden_channels, "add_self_loops": False}
            last_kwargs = {"in_channels": (-1, -1), "out_channels": out_channels, "add_self_loops": False}
        elif args.layer == "SAGEConv":
            layer = SAGEConv
            kwargs = {"in_channels": (-1, -1), "out_channels": hidden_channels, "add_self_loops": False}
            last_kwargs = {"in_channels": (-1, -1), "out_channels": out_channels, "add_self_loops": False}
        elif args.layer == "GraphConv":
            layer = GraphConv
            kwargs = {"in_channels": (-1, -1), "out_channels": hidden_channels, "add_self_loops": False}
            last_kwargs = {"in_channels": (-1, -1), "out_channels": out_channels, "add_self_loops": False}
        elif args.layer == "LEConv":
            layer = LEConv
            kwargs = {"in_channels": (-1, -1), "out_channels": hidden_channels, "add_self_loops": False}
            last_kwargs = {"in_channels": (-1, -1), "out_channels": out_channels, "add_self_loops": False}
        else:
            raise NotImplementedError(f"Layer {args.layer} is not implemented")

        if args.num_layer <= 1:
            self.conv_layers = nn.ModuleList([layer(**last_kwargs)])
            self.lin_layers = nn.ModuleList([Linear(-1, out_channels)])
        else:
            self.conv_layers = nn.ModuleList([layer(**kwargs)])
            self.lin_layers = nn.ModuleList([Linear(-1, hidden_channels)])
            for i in range(args.num_layer - 2):
                self.conv_layers.append(layer(**kwargs))
                self.lin_layers.append(Linear(-1, hidden_channels))

            self.conv_layers.append(layer(**last_kwargs))
            self.lin_layers.append(Linear(-1, out_channels))

    def forward(self, x, edge_index, edge_attr, edge_weight):
        kwargs = {"edge_index": edge_index}
        if self.args.layer == "GATConv":
            kwargs["edge_attr"] = edge_attr
        elif self.args.layer == "LEConv":
            kwargs["edge_weight"] = edge_weight

        for conv, lin in zip(self.conv_layers[:-1], self.lin_layers[:-1]):
            x = conv(x, **kwargs) + lin(x)
            x = x.relu()
        x = self.conv_layers[-1](x, **kwargs) + self.lin_layers[-1](x)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(
            self,
            z_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
    ):
        # row, col = edge_label_index
        # z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        # z_ptp is empty
        # z_ptp = torch.cat(
        #     [
        #         z_dict["pnode"][ptp_edge_label_index[0]],
        #         z_dict["pnode"][ptp_edge_label_index[1]],
        #     ],
        #     dim=-1,
        # )
        z_ptnp = torch.cat(
            [
                z_dict["pnode"][ptnp_edge_label_index[0]],
                z_dict["onode"][ptnp_edge_label_index[1]],
            ],
            dim=-1,
        )
        z_nptp = torch.cat(
            [
                z_dict["onode"][nptp_edge_label_index[0]],
                z_dict["pnode"][nptp_edge_label_index[1]],
            ],
            dim=-1,
        )
        z_nptnp = torch.cat(
            [
                z_dict["onode"][nptnp_edge_label_index[0]],
                z_dict["onode"][nptnp_edge_label_index[1]],
            ],
            dim=-1,
        )

        z = torch.cat([z_ptnp, z_nptp, z_nptnp], dim=0)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class LiquorGNNModel(torch.nn.Module):
    def __init__(self, metadata, args):
        super().__init__()
        self.encoder = GNNEncoder(args.hidden_channels, args.hidden_channels, args)
        self.encoder = to_hetero(self.encoder, metadata, aggr="sum")
        self.decoder = EdgeDecoder(args.hidden_channels)

    def forward(
            self,
            x_dict,
            edge_index_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
            attrs,
            edge_weight_dict
    ):
        z_dict = self.encoder(x_dict, edge_index_dict, attrs, edge_weight_dict)
        return self.decoder(
            z_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
        )


class LiquorGNNModelPL(pl.LightningModule):
    def __init__(self, metadata, args) -> None:
        super().__init__()
        self.model = LiquorGNNModel(metadata, args)
        self.save_hyperparameters()
        self.learning_rate = 1e-3 if args.lr == "auto" else float(args.lr)
        self.batch_size = args.batch_size
        prefix = generate_date_prefix()
        random.seed(args.seed)
        self.args = args
        self.prefix = os.path.dirname(
            os.path.abspath(__file__)) + f'/../results/{args.layer}/{prefix}'
        Path(self.prefix).mkdir(parents=True, exist_ok=True)
        self.predictions = []
        self.logits = []
        self.labels = []
        self.val_predictions = []
        self.val_logits = []
        self.val_labels = []

    def create_mapping(self, test_ds):
        df_pressures = pd.read_csv("dataset/2018_SCADA_Pressures.csv", delimiter=";", decimal=",")
        df_demands = pd.read_csv("dataset/predicted_demands_2018.zip", compression="zip")
        nodes_with_pressures = df_pressures.columns[1:]
        nodes_without_pressures = [
            n for n in df_demands.columns[2:] if n not in nodes_with_pressures
        ]
        inp_file = "dataset/L-TOWN.inp"
        wn = wntr.network.WaterNetworkModel(inp_file)
        G = wn.get_graph()

        dict_ID_to_Node = dict()
        i = 0
        for n in G.nodes():
            G.nodes[n]["ID"] = i
            dict_ID_to_Node[i] = n
            i += 1

        num_nodes = len(nodes_with_pressures) + len(nodes_without_pressures)
        ID_pressure = [
            id for id in range(num_nodes) if not dict_ID_to_Node[id] in nodes_without_pressures
        ]
        ID_no_pressure = [
            id for id in range(num_nodes) if not dict_ID_to_Node[id] in nodes_with_pressures
        ]

        # Mapping from output id/order to pipe
        out_to_pipe = {}
        pipe_to_out = {}
        idx = 0
        for i in range(len(test_ds.data.edge_stores)):
            for j in range(len(test_ds.data.edge_stores[i].edge_index[0,
                               :test_ds.data.edge_stores[i].edge_index.shape[1] // len(test_ds)])):
                if test_ds.data.edge_types[i][0] == 'pnode':
                    id1 = ID_pressure[test_ds.data.edge_stores[i].edge_index[0][j]]
                else:
                    id1 = ID_no_pressure[test_ds.data.edge_stores[i].edge_index[0][j]]

                if test_ds.data.edge_types[i][2] == 'pnode':
                    id2 = ID_pressure[test_ds.data.edge_stores[i].edge_index[1][j]]
                else:
                    id2 = ID_no_pressure[test_ds.data.edge_stores[i].edge_index[1][j]]
                out_to_pipe[idx] = get_link(dict_ID_to_Node[id1], dict_ID_to_Node[id2], wn)
                pipe_to_out[get_link(dict_ID_to_Node[id1], dict_ID_to_Node[id2], wn).name] = idx
                idx += 1

        self.out_to_pipe = out_to_pipe
        self.pipe_to_out = pipe_to_out

    def forward(
            self,
            x_dict,
            edge_index_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
            attrs,
            weights
    ):
        return self.model(
            x_dict,
            edge_index_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
            attrs,
            weights
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def process_batch(self, batch):
        x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
        ptnp_edges = batch["pnode", "connects", "onode"]
        nptp_edges = batch["onode", "connects", "pnode"]
        nptnp_edges = batch["onode", "connects", "onode"]

        # ptp_edge_label_index = (
        #     ptp_edges.edge_index
        #     if len(ptp_edges) > 0
        #     else torch.tensor([[], []], device=self.device)
        # )
        ptnp_edge_label_index = (
            ptnp_edges.edge_index
            if len(ptnp_edges) > 0
            else torch.tensor([[], []], device=self.device)
        )
        nptp_edge_label_index = (
            nptp_edges.edge_index
            if len(nptp_edges) > 0
            else torch.tensor([[], []], device=self.device)
        )
        nptnp_edge_label_index = (
            nptnp_edges.edge_index
            if len(nptnp_edges) > 0
            else torch.tensor([[], []], device=self.device)
        )

        attrs = batch.edge_attr_dict
        weights = batch.edge_weight_dict

        return x_dict, edge_index_dict, ptnp_edge_label_index, nptp_edge_label_index, nptnp_edge_label_index, attrs, weights

    def training_step(self, train_batch: HeteroData, batch_idx):
        x_dict, edge_index_dict, ptnp_edge_label_index, nptp_edge_label_index, nptnp_edge_label_index, attrs, weights = self.process_batch(
            train_batch)

        y_hat = self.forward(
            x_dict,
            edge_index_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
            attrs,
            weights
        )
        y = train_batch.collect('edge_label')

        con = []
        for key in y.keys():
            con.append(y[key])
        y = torch.cat(con).to(torch.float32)

        loss = binary_cross_entropy_with_logits(y_hat, y)

        return loss

    def validation_step(self, val_batch: HeteroData, batch_idx):
        x_dict, edge_index_dict, ptnp_edge_label_index, nptp_edge_label_index, nptnp_edge_label_index, attrs, weights = self.process_batch(
            val_batch)

        y_hat = self.forward(
            x_dict,
            edge_index_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
            attrs,
            weights
        )
        y = val_batch.collect('edge_label')

        con = []
        for key in y.keys():
            con.append(y[key])
        y = torch.cat(con).to(torch.float32)

        loss = binary_cross_entropy_with_logits(y_hat, y)

        self.val_predictions.append(torch.where(y_hat > 0, True, False))
        self.val_logits.append(y_hat)
        self.val_labels.append(y)

        return loss


    def calc_custom_eco_score(self, logits, labels, detection_idxs, detections):
        inp_file = "dataset/L-TOWN.inp"
        wn = wntr.network.WaterNetworkModel(inp_file)
        G = wn.get_graph()

        df = pd.DataFrame(np.vstack((detection_idxs, detections)).T, columns=["Id", "Pipe"])
        df = df.sort_values(by=['Id']).reset_index(drop=True)

        def find_close_pipes(pipe, wn):
            pipe = wn.links[pipe]

            close_pipes = [pipe]
            pipe_dist = [0]

            old_pipes = [pipe]
            nodes = [pipe.start_node, pipe.end_node]
            l = pipe.length if hasattr(pipe, "length") else 10
            distances = [l / 2, l / 2]
            while len(nodes) > 0:
                node = nodes.pop(0)
                dist = distances.pop(0)
                poss_pipes = [list(x)[0] for x in list({**G.pred[node.name], **G.succ[node.name]}.values())]
                for p in poss_pipes:
                    p = wn.links[p]
                    length = p.length if hasattr(p, "length") else 10
                    if (p not in old_pipes or dist + length / 2 < pipe_dist[
                        close_pipes.index(p)]) and dist + length / 2 < 300:
                        if p in old_pipes:
                            pipe_dist[close_pipes.index(p)] = dist + length / 2
                        else:
                            old_pipes.append(p)
                            close_pipes.append(p)
                            pipe_dist.append(dist + length / 2)

                        if p.start_node == node:
                            nodes.append(p.end_node)
                        else:
                            nodes.append(p.start_node)
                        distances.append(dist + length)

            return [x.name for x in close_pipes], pipe_dist

        score = 0
        already_detected = []
        for idx, pipe in zip(df["Id"], df["Pipe"]):
            possible_det, distances = find_close_pipes(pipe, wn)
            if "p239" in possible_det:
                distances.pop(possible_det.index("p239"))
                possible_det.pop(possible_det.index("p239"))
            if "p235" in possible_det:
                distances.pop(possible_det.index("p235"))
                possible_det.pop(possible_det.index("p235"))
            if "p227" in possible_det:
                distances.pop(possible_det.index("p227"))
                possible_det.pop(possible_det.index("p227"))
            if "PUMP_1" in possible_det:
                distances.pop(possible_det.index("PUMP_1"))
                possible_det.pop(possible_det.index("PUMP_1"))
            pipe_detected_leak = False
            for det, dist in zip(possible_det, distances):
                if labels[int(idx), self.pipe_to_out[det]]:
                    if det not in already_detected:
                        # true detection
                        s = (1 * 1 * labels[int(idx):, self.pipe_to_out[det]].sum()) - 500 * (dist / 300)
                        score += s
                        already_detected.append(det)
                        pipe_detected_leak = True
                        break
            if not pipe_detected_leak:
                score -= 500

        return score

    def validation_epoch_end(self, outputs) -> None:
        all_labels = []
        all_logits = []
        for i in range(len(self.val_logits)):
            batches_no = len(self.val_logits[i]) // 905
            hetero_total = [x * batches_no for x in [30, 39, 836]]
            lab = []
            log = []
            s = 0
            for j in range(3):
                lab.append(self.val_labels[i][s: s + hetero_total[j]].cpu().detach().numpy())
                log.append(self.val_logits[i][s: s + hetero_total[j]].cpu().detach().numpy())
                s += hetero_total[j]
    
            labels_np = np.concatenate([x.reshape(batches_no, -1) for x in lab], 1)
            logits = np.concatenate([x.reshape(batches_no, -1) for x in log], 1)
            all_labels.append(labels_np)
            all_logits.append(logits)
        
        labels_np = np.vstack(all_labels)
        logits = np.vstack(all_logits)

        labels_np = np.where(labels_np > 0.5, True, False)

        thresholds = [0, -0.25, -0.5, -0.75, -1, -1.5, -2, -2.5, -3, -4, -5, -6, -7, -8, -9, -10, -12.5, -15]
        precisions = []
        recalls = []
        f1s = []
        tp2s = []
        fp2s = []
        fn2s = []
        recalls2 = []
        f1s2 = []
        eco_scores = []

        for thresh in thresholds:
            predictions = np.where(logits > thresh, True, False)

            tn, fp, fn, tp = confusion_matrix(labels_np.reshape(-1), predictions.reshape(-1), labels=[0, 1]).ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            idx = np.any(predictions != False, axis=0)
            predictions2 = predictions[:, idx]
            columns = np.nonzero(idx)[0]
            df = pd.DataFrame(predictions2, columns=list(columns))
            pipe_names = [self.out_to_pipe[x] for x in df.columns]

            detections = []
            detection_idxs = []

            pred = df.to_numpy()
            tp2, fp2, fn2 = 0, 0, 0
            out = "# linkID, startTime\n"
            for i in range(pred.shape[1] - 1):
                true_count = 0
                false_count = 0
                enough_false = True
                for j in range(len(pred)):
                    if pred[j, i]:
                        true_count += 1
                        false_count = 0
                    else:
                        true_count = 0
                        false_count += 1
                    if true_count == 5 and enough_false:                        # 5 times True in a row: writes leakage in out.txt
                        out += f"{pipe_names[i]}\n"
                        detections.append(pipe_names[i].name)
                        detection_idxs.append(j - 4)
                        enough_false = False
                        if labels_np[j-3, columns[i]]:
                            tp2 += 1
                            break  # Pipe can only be reported once if it was reported correctly
                        else:
                            fp2 += 1
                    elif false_count == 2016:                                   # After 7 (12 * 24 * 7 = 2016) days of False, pipes can be written into out.txt again
                        enough_false = True

            eco_score = self.calc_custom_eco_score(logits, labels_np, detection_idxs, detections)
            eco_scores.append(eco_score)

            fn2 = len(np.where(np.any(labels_np != False, axis=0))[0]) - tp2
            precision2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
            recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
            f12 = 2 * precision2 * recall2 / (precision2 + recall2) if (precision2 + recall2) > 0 else 0
            tp2s.append(tp2)
            fp2s.append(fp2)
            fn2s.append(fn2)
            recalls2.append(recall2)
            f1s2.append(f12)

        val_df = pd.DataFrame(np.vstack((thresholds, precisions, recalls, f1s, tp2s, fp2s, fn2s, recalls2, f1s2, eco_scores)).T,
                              columns=["Threshold", "Precision", "Recall", "F1", "tp2", "fp2", "fn2", "Recall2", "F12", "Score"])

        # print()
        # print(val_df)

        max_score_df = val_df[val_df["Score"] == val_df["Score"].max()]
        best_thresh_df = max_score_df[max_score_df["Recall"] == max_score_df["Recall"].max()]

        self.val_auc = roc_auc_score(labels_np.reshape(-1), logits.reshape(-1))
        self.val_prec = best_thresh_df["Precision"].values[0]
        self.val_recall = best_thresh_df["Recall"].values[0]
        self.val_f1 = best_thresh_df["F1"].values[0]

        self.best_threshold = best_thresh_df["Threshold"].values[0]

        self.val_predictions = []
        self.val_logits = []
        self.val_labels = []

        return self.log("loss/val", np.mean([x.cpu() for x in outputs]))  # Switch outputs to cpu if model is trained on gpu

    def test_step(self, test_batch: HeteroData, batch_idx):
        x_dict, edge_index_dict, ptnp_edge_label_index, nptp_edge_label_index, nptnp_edge_label_index, attrs, weights = self.process_batch(
            test_batch)

        y_hat = self.forward(
            x_dict,
            edge_index_dict,
            # ptp_edge_label_index,
            ptnp_edge_label_index,
            nptp_edge_label_index,
            nptnp_edge_label_index,
            attrs,
            weights
        )
        y = test_batch.collect('edge_label')

        con = []
        for key in y.keys():
            con.append(y[key])
        y = torch.cat(con)  # .to(torch.float32)

        # self.predictions.append(torch.where(y_hat > self.best_threshold, True, False))
        self.predictions.append(torch.where(y_hat > -10, True, False))
        self.logits.append(y_hat)
        self.labels.append(y)

    def test_epoch_end(self, outputs) -> None:
        predictions = torch.hstack(self.predictions)
        self.labels = torch.hstack(self.labels)
        predictions = predictions.cpu().detach().numpy()
        labels2 = self.labels.cpu().detach().numpy()
        labels2 = np.where(labels2 > 0.5, True, False)
        tn, fp, fn, tp = confusion_matrix(labels2, predictions).ravel()

        all_predictions = []
        all_logits = []
        for i in range(len(self.logits)):
            batches_no = len(self.logits[i]) // 905
            hetero_total = [x * batches_no for x in [30, 39, 836]]
            pred = []
            log = []
            s = 0
            for j in range(3):
                pred.append(self.predictions[i][s: s + hetero_total[j]].cpu().detach().numpy())
                log.append(self.logits[i][s: s + hetero_total[j]].cpu().detach().numpy())
                s += hetero_total[j]

            pred = np.concatenate([x.reshape(batches_no, -1) for x in pred], 1)
            log = np.concatenate([x.reshape(batches_no, -1) for x in log], 1)
            all_predictions.append(pred)
            all_logits.append(log)

        predictions = np.vstack(all_predictions)
        logits = np.vstack(all_logits)

        labels = pd.read_csv("dataset/2019_Leakages.csv", delimiter=";")

        idx = np.any(predictions != False, axis=0)
        predictions2 = predictions[:, idx]

        columns = np.nonzero(idx)[0]
        df = pd.DataFrame(predictions2, columns=list(columns))
        df["Timestamp"] = labels["Timestamp"]

        names = []
        for idx, column in enumerate(df):
            if column in self.out_to_pipe:
                names.append(self.out_to_pipe[column].name)
        names.append("Timestamp")
        df.columns = names

        perfect_team = pd.read_csv("dataset/LiquorGNN_out.csv")
        pt = perfect_team.to_numpy()[:, 1:].T

        pred = df.to_numpy()
        tp2, fp2, fn2 = 0, 0, 0
        out = "# linkID, startTime\n"
        for i in range(pred.shape[1] - 1):
            true_count = 0
            false_count = 0
            enough_false = True
            for j in range(len(pred)):
                if pred[j, i]:
                    true_count += 1
                    false_count = 0
                else:
                    true_count = 0
                    false_count += 1
                if true_count == 5 and enough_false:  # 5 times True in a row: writes leakage in out.txt
                    out += f"{df.columns[i]}, {df.values[j - 3, -1][:-3]}\n"
                    enough_false = False
                    if df.columns[i] in perfect_team["linkID"].values and pt[1, np.where(pt[0] == df.columns[i])][
                        0, 0] <= pred[j - 3, -1][:-2]:
                        tp2 += 1
                        break  # Pipe can only be reported once if it was reported correctly
                    else:
                        fp2 += 1
                elif false_count == 2016:  # After 7 (12 * 24 * 7 = 2016) days of False, pipes can be written into out.txt again
                    enough_false = True

        fn2 = len(perfect_team) - tp2

        with open(self.prefix + '/out.txt', 'w') as f:
            f.write(f"TP: {tp}\n")
            f.write(f"TN: {tn}\n")
            f.write(f"FP: {fp}\n")
            f.write(f"FN: {fn}\n\n")
            f.write(out)
            f.write(f"\nTP: {tp2}\n")
            f.write(f"FP: {fp2}\n")
            f.write(f"TN: {fn2}\n\n")

        self.plot_network(df, labels, predictions)

    def plot_network(self, pred, labels, full_pred):
        # plot leakages from 2019 and predicted leakages
        inp_file = "dataset/L-TOWN.inp"
        wn = wntr.network.WaterNetworkModel(inp_file)
        pressures = pd.read_csv("dataset/2019_SCADA_Pressures.csv", delimiter=";", decimal=",")

        names = []
        for column in range(full_pred.shape[1]):
            if column in self.out_to_pipe:
                names.append(self.out_to_pipe[column].name)

        leak = {}
        for col in names:
            if col == "Timestamp":
                continue
            if col in pred.columns and col in labels.columns:
                leak[col] = 3.0  # true positive
            elif col in pred.columns:
                leak[col] = 2.0  # false positive
            elif col in labels.columns:
                leak[col] = 1.0  # false negative
            else:
                leak[col] = 0.0  # true negative

        pressure = {node: 1.0 if node in pressures else 0.0 for node in wn.nodes}

        wn2 = wntr.morph.convert_node_coordinates_to_longlat(wn, {"n1": (-32.40332, 52.51815),
                                                                  "n347": (-32.40338, 52.51877)})
        colors = ['blue', "red", "orange", 'green']
        node_colors = ["white", "black"]
        wntr.graphics.network.plot_leaflet_network(wn2, node_attribute=pressure, link_attribute=leak, link_width=3,
                                                   add_legend=True,
                                                   node_attribute_name="Pressure", link_attribute_name="Leakage",
                                                   node_size=3,
                                                   node_cmap=node_colors, link_cmap=colors, link_cmap_bins='cut',
                                                   filename=self.prefix + '/network.html')
