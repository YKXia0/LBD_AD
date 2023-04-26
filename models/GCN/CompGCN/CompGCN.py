import argparse
from time import time

import numpy as np
import pandas as pd
import torch as th
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_loader import Data
from models import LinkPredict
from utils import in_out_norm


def predict(model, graph, device, data_iter, split="valid", mode="tail", neg_sample_size_eval=20, eval_p="filtered"):
    model.eval()
    with th.no_grad():
        results = {}
        train_iter = iter(data_iter["{}_{}".format(split, mode)])

        sub = th.tensor([], dtype=th.long).to(device)
        rel = th.tensor([], dtype=th.long).to(device)
        obj = th.tensor([], dtype=th.long).to(device)

        for step, batch in enumerate(train_iter):
            triple = batch[0].to(device)
            sub_batch, rel_batch, obj_batch = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
            )

            sub = th.cat([sub, sub_batch], 0)
            rel = th.cat([rel, rel_batch], 0)
            obj = th.cat([obj, obj_batch], 0)

        pred = model(graph, sub, rel)

        b_range = th.arange(pred.size()[0], device=device)
        target_pred = pred[b_range, obj]
        pred[b_range, obj] = target_pred

        # Negative sampling
        num_nodes = pred.shape[1]
        neg_sample = th.tensor([], dtype=th.long).to(device)
        for index, i in enumerate(obj):
            tmp = th.tensor([i])
            tmp = th.cat([tmp, th.randperm(num_nodes)[:(neg_sample_size_eval - 1)]], 0)
            if index == 0:
                neg_sample = th.hstack((neg_sample, tmp.to(device)))
            else:
                neg_sample = th.vstack((neg_sample, tmp.to(device)))

        sample_pred = []
        for i in range(len(neg_sample)):
            tmp = []
            for j in range(len(neg_sample[0])):
                tmp.append(float(pred[i][int(neg_sample[i][j])]))

            sample_pred.append(tmp)

        sample_pred = th.tensor(sample_pred).to(device)

        ranks = (
                1
                + th.argsort(
            th.argsort(sample_pred, dim=1, descending=True),
            dim=1,
            descending=False,
        )[b_range, 0]
        )

        ranks = ranks.float()
        results["count"] = th.numel(ranks) + results.get("count", 0.0)
        results["mr"] = th.sum(ranks).item() + results.get("mr", 0.0)
        results["mrr"] = th.sum(1.0 / ranks).item() + results.get(
            "mrr", 0.0
        )
        for k in [1, 3, 10]:
            results["hits@{}".format(k)] = th.numel(
                ranks[ranks <= (k)]
            ) + results.get("hits@{}".format(k), 0.0)

    return results


# evaluation function, evaluate the head and tail prediction and then combine the results
def evaluate(model, graph, device, data_iter, split="valid", neg_sample_size_eval=20, eval_p="filtered"):
    # predict for head and tail
    left_results = predict(model, graph, device, data_iter, split, "tail", neg_sample_size_eval, eval_p)
    right_results = predict(model, graph, device, data_iter, split, "head", neg_sample_size_eval, eval_p)
    results = {}
    count = float(left_results["count"])

    # combine the head and tail prediction results
    # Metrics: MRR, MR, and Hit@k
    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round(
        (left_results["mr"] + right_results["mr"]) / (2 * count), 5
    )
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )
    for k in [1, 3, 10]:
        results["left_hits@{}".format(k)] = round(
            left_results["hits@{}".format(k)] / count, 5
        )
        results["right_hits@{}".format(k)] = round(
            right_results["hits@{}".format(k)] / count, 5
        )
        results["hits@{}".format(k)] = round(
            (
                    left_results["hits@{}".format(k)]
                    + right_results["hits@{}".format(k)]
            )
            / (2 * count),
            5,
        )
    return results


def main(args):
    # Split data
    g = pd.read_csv(args.dataset, header=None, sep="\t")
    g = g[[0, 1, 2]]
    trainDF, validDF = train_test_split(g, test_size=0.2, shuffle=False)
    validDF, testDF = train_test_split(validDF, test_size=0.5, shuffle=False)

    trainDF.to_csv("train.txt", header=None, index=None, sep="\t")
    validDF.to_csv("valid.txt",
                   header=None, index=None, sep="\t")
    testDF.to_csv("test.txt", header=None, index=None, sep="\t")

    # Prepare graph data and retrieve train/validation/test index
    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # construct graph, split in/out edges and prepare train/validation/test data_loader
    data = Data(
        args.lbl_smooth, args.num_workers, args.batch_size
    )
    data_iter = data.data_iter  # train/validation/test data_loader

    graph = data.g.to(device)
    num_rel = th.max(graph.edata["etype"]).item() + 1

    # Compute in/out edge norms and store in edata
    graph = in_out_norm(graph)

    # Step 2: Create model
    compgcn_model = LinkPredict(
        num_bases=args.num_bases,
        num_rel=num_rel,
        num_ent=graph.num_nodes(),
        in_dim=args.init_dim,
        layer_size=args.layer_size,
        comp_fn=args.opn,
        batchnorm=True,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        num_filt=args.num_filt,
        hid_drop=args.hid_drop,
        feat_drop=args.feat_drop,
        ker_sz=args.ker_sz,
        k_w=args.k_w,
        k_h=args.k_h,
    )
    compgcn_model = compgcn_model.to(device)

    # Create training components
    loss_fn = th.nn.BCELoss()
    optimizer = optim.Adam(
        compgcn_model.parameters(), lr=args.lr, weight_decay=args.l2
    )

    # training epochs
    for epoch in range(args.max_epochs):
        # Training and validation using a full graph
        compgcn_model.train()
        train_loss = []
        t0 = time()
        for step, batch in enumerate(data_iter["train"]):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            logits = compgcn_model(graph, sub, rel)

            # compute loss
            tr_loss = loss_fn(logits, label)
            train_loss.append(tr_loss.item())

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        train_loss = np.sum(train_loss)
        t1 = time()

        print(
            "In epoch {}, Train Loss: {:.4f}, Train time: {}".format(
                epoch, train_loss, t1 - t0
            ))

    th.save(
        compgcn_model.state_dict(), "comp_link" + "_model_state.pth"
    )

    # test; calculate the metrics
    compgcn_model.eval()
    compgcn_model.load_state_dict(th.load("comp_link" + "_model_state.pth"))
    test_results = evaluate(
        compgcn_model, graph, device, data_iter, "test", 20, "filtered"
    )

    print(
        "Test MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
            test_results["mrr"],
            test_results["mr"],
            test_results["hits@10"],
            test_results["hits@3"],
            test_results["hits@1"],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="dataset",
        default="graph.tsv",
        help="Dataset to use",
    )

    parser.add_argument(
        "--score_func",
        dest="score_func",
        default="conve",
        help="Score Function for Link prediction",
    )

    parser.add_argument(
        "--opn",
        dest="opn",
        default="ccorr",
        help="Composition Operation to be used in CompGCN",
    )

    parser.add_argument(
        "--batch", dest="batch_size", default=1024, type=int, help="Batch size"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default="0",
        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
    )

    parser.add_argument(
        "--epoch",
        dest="max_epochs",
        type=int,
        default=30,
        help="Number of epochs",
    )

    parser.add_argument(
        "--l2", type=float, default=0.0, help="L2 Regularization for Optimizer"
    )

    parser.add_argument(
        "--lr", type=float, default=0.001, help="Starting Learning Rate"
    )

    parser.add_argument(
        "--lbl_smooth",
        dest="lbl_smooth",
        type=float,
        default=0.1,
        help="Label Smoothing",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of processes to construct batches",
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        default=41504,
        type=int,
        help="Seed for randomization",
    )

    parser.add_argument(
        "--num_bases",
        dest="num_bases",
        default=-1,
        type=int,
        help="Number of basis relation vectors to use",
    )

    parser.add_argument(
        "--init_dim",
        dest="init_dim",
        default=100,
        type=int,
        help="Initial dimension size for entities and relations",
    )

    parser.add_argument(
        "--layer_size",
        nargs="?",
        default="[200]",
        help="List of output size for each compGCN layer",
    )

    parser.add_argument(
        "--gcn_drop",
        dest="dropout",
        default=0.1,
        type=float,
        help="Dropout to use in GCN Layer",
    )

    parser.add_argument(
        "--layer_dropout",
        nargs="?",
        default="[0.3]",
        help="List of dropout value after each compGCN layer",
    )

    parser.add_argument(
        "--hid_drop",
        dest="hid_drop",
        default=0.3,
        type=float,
        help="ConvE: Hidden dropout",
    )

    parser.add_argument(
        "--feat_drop",
        dest="feat_drop",
        default=0.3,
        type=float,
        help="ConvE: Feature Dropout",
    )

    parser.add_argument(
        "--k_w", dest="k_w", default=10, type=int, help="ConvE: k_w"
    )

    parser.add_argument(
        "--k_h", dest="k_h", default=20, type=int, help="ConvE: k_h"
    )

    parser.add_argument(
        "--num_filt",
        dest="num_filt",
        default=200,
        type=int,
        help="ConvE: Number of filters in convolution",
    )

    parser.add_argument(
        "--ker_sz",
        dest="ker_sz",
        default=7,
        type=int,
        help="ConvE: Kernel size to use",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    print(args)

    args.layer_size = eval(args.layer_size)
    args.layer_dropout = eval(args.layer_dropout)

    main(args)
