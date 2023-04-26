import argparse
import numpy as np
import pandas as pd
import torch

import utils
from model import LinkPredict
from data_loader import Data


def main(args):
    graph = args.graph
    triples = pd.read_csv(graph, sep='\t')
    triples.sort_values("PMID", ascending=True, inplace=True, ignore_index=True)
    triples = triples[triples.columns[:3]]
    knowledge_graph = Data(triples)
    num_nodes, num_rels, num_edges = knowledge_graph.get_stats()
    print('# entities:', num_nodes)
    print('# relations:', num_rels)
    print('# edges:', num_edges)

    train_data_np, valid_data_np, test_data_np = knowledge_graph.get_train_test_data(0.1, 0.1)
    train_data_np = np.concatenate((train_data_np, valid_data_np))
    # train = torch.LongTensor(train_data_np)
    # valid_data = torch.LongTensor(valid_data_np)  # no valid
    test_data = torch.LongTensor(test_data_np)
    total_data = torch.LongTensor(np.concatenate((train_data_np, test_data)))

    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.num_bases,
                        num_hidden_layers=args.num_hidden_layers,
                        dropout=args.dropout,
                        use_cuda=args.use_cuda,
                        reg_param=args.reg_param)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = model.to(device)
    print(device)

    # build train graph
    train_graph, train_rel, train_norm = utils.build_graph(num_nodes, num_rels, train_data_np)
    train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_graph(num_nodes, num_rels, test_data_np)
    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = utils.node_norm_2_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    # build adj list and calculate degrees for sampling
    adj_list = utils.get_adj(num_nodes, train_data_np)  # degrees

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("start training...")

    # epoch is step
    for iteration in range(1, 1 + args.iterations):
        model.train()

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data_np, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, train_deg, args.negative_sample,
                args.edge_sampler)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = utils.node_norm_2_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

        # Load on devcie
        g = g.to(device)
        node_id = node_id.to(device)
        edge_type = edge_type.to(device)
        edge_norm = edge_norm.to(device)
        data = data.to(device)
        labels = labels.to(device)

        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()

        if iteration % args.evaluate_every == 0:
            print("Epoch {} | Loss {:.5f}".format(iteration, loss.item()))

        optimizer.zero_grad()

    torch.save({'state_dict': model.state_dict(), 'iteration': iteration}, args.model_state_file)

    print("Evaluating...")
    model.eval()
    test_data = torch.LongTensor(test_data_np)
    test_graph = test_graph.to(device)
    test_node_id = test_node_id.to(device)
    test_rel = test_rel.to(device)
    test_norm = test_norm.to(device)
    test_data = test_data.to(device)
    total_data = torch.LongTensor(total_data)

    output = model(test_graph, test_node_id, test_rel, test_norm)

    import time
    old_time = time.time()

    hits = [1, 3, 10]
    mr, mrr, hits_dict = utils.calc_mrr(output, model.w_relation, test_data,
                                        torch.LongTensor(total_data).to(device),
                                        batch_size=args.eval_batch_size, neg_sample_size_eval=args.neg_sample_size_eval,
                                        hits=hits, eval_p=args.eval_protocol)

    new_time = time.time()
    print(new_time - old_time)

    print(f"MR: {mr:.6f}")
    print(f"MRR: {mrr:.6f}")
    for key, value in hits_dict.items():
        print(f"Hits @ {key} = {value:.6f}")

    print("Training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="graph",
        default="graph.tsv",
        help="Dataset to use",
    )

    parser.add_argument(
        "--n_hidden",
        dest="n_hidden",
        type=int,
        default=200,
        help="Dimensions of the hidden layer",
    )

    parser.add_argument(
        "--num_bases",
        dest="num_bases",
        type=int,
        default=20,
        help="Number of basis relation vectors to use",
    )

    parser.add_argument(
        "--num_hidden_layers",
        dest="num_hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers",
    )

    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        default=0.2,
        help="Number of hidden layers",
    )

    parser.add_argument(
        "--use_cuda",
        dest="use_cuda",
        type=bool,
        default=True,
        help="GPU",
    )

    parser.add_argument(
        "--reg_param",
        dest="reg_param",
        type=float,
        default=0.01,
        help="GPU",
    )


    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=80000,
        help="Number of iterations (iterations = epochs * (datasize / batchsize))",
    )

    parser.add_argument(
        "--evaluate_every",
        dest="evaluate_every",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--graph_batch_size",
        dest="graph_batch_size",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--graph_split_size",
        dest="graph_split_size",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--negative_sample",
        dest="negative_sample",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--edge_sampler",
        dest="edge_sampler",
        default="uniform",
    )

    parser.add_argument(
        "--grad_norm",
        dest="grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--neg_sample_size_eval",
        dest="neg_sample_size_eval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval_protocol",
        dest="eval_protocol",
        default="filtered",
    )

    parser.add_argument(
        "--model_state_file",
        dest="model_state_file",
        default="model_state.pth",
    )

    args = parser.parse_args()

    main(args)