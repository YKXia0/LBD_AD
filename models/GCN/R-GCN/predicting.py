import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import torch

import utils
from model import LinkPredict


class KnowledgeGraph():
    def __init__(self, graph):
        self.df_graph = graph.copy()
        self.generate_dictionary()
        self.total_data = self.generate_dataset()
        self.num_nodes, self.num_rels, self.num_edges = self.get_stats()

    def generate_dictionary(self):
        self.entity2index = {}
        self.relation2index = {}

        self.index2entity = {}
        self.index2relation = {}

        entity_index = 0
        relation_index = 0

        for index, triple in self.df_graph.iterrows():

            # entity - index
            if triple[0] not in self.entity2index:
                self.entity2index[triple[0]] = entity_index
                self.index2entity[entity_index] = triple[0]
                entity_index += 1

            if triple[2] not in self.entity2index:
                self.entity2index[triple[2]] = entity_index
                self.index2entity[entity_index] = triple[2]
                entity_index += 1

            # relation - index
            # (ignore the types of entities)
            relation = triple[1]
            if ('head', relation, 'tail') not in self.relation2index:
                self.relation2index[('head', relation, 'tail')] = relation_index
                self.index2relation[relation_index] = ('head', relation, 'tail')
                relation_index += 1

    def generate_dataset(self):
        # Transfer name to index in the dataset
        idtrpile_list = []
        for index, triple in self.df_graph.iterrows():
            idtrpile = []

            idtrpile.append(self.entity2index[triple[0]])
            idtrpile.append(self.relation2index[('head', triple[1], 'tail')])
            idtrpile.append(self.entity2index[triple[2]])

            idtrpile_list.append(idtrpile)

        total_data = np.asarray(idtrpile_list)

        return total_data

    def get_stats(self):
        num_nodes = len(self.entity2index)
        num_rels = len(self.relation2index)
        num_edges = len(self.df_graph)
        print('# entities:', num_nodes)
        print('# relations:', num_rels)
        print('# edges:', num_edges)

        return num_nodes, num_rels, num_edges


    def create_model(self, n_hidden, num_bases, num_hidden_layers, dropout, use_cuda, reg_param):
        model = LinkPredict(self.num_nodes,
                            n_hidden,
                            self.num_rels,
                            num_bases=num_bases,
                            num_hidden_layers=num_hidden_layers,
                            dropout=dropout,
                            use_cuda=use_cuda,
                            reg_param=reg_param)
        return model


    def load_model(self):

        model = self.create_model(args.n_hidden, args.num_bases, args.num_hidden_layers, args.dropout, args.use_cuda, args.reg_param)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')

        model = model.to(device)
        print(device)

        # use best model checkpoint
        checkpoint = torch.load(args.model_state_file)
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])

        # build total graph
        total_graph, total_rel, total_norm = utils.build_graph(self.num_nodes, self.num_rels, self.total_data)
        total_node_id = torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1)
        total_rel = torch.from_numpy(total_rel)
        total_norm = utils.node_norm_2_edge_norm(total_graph, torch.from_numpy(total_norm).view(-1, 1))

        total_graph = total_graph.to(device)
        total_node_id = total_node_id.to(device)
        total_rel = total_rel.to(device)
        total_norm = total_norm.to(device)

        # get embed weights
        self.total_embed = model(total_graph, total_node_id, total_rel, total_norm)
        self.w = model.w_relation

    def predict_score_with_s_r_o(self, s_index, r_index, o_index):
        emb_s = self.total_embed[s_index]
        emb_r = self.w[r_index]
        emb_o = self.total_embed[o_index]

        emb_triplet = emb_s * emb_r * emb_o
        score = torch.sigmoid(torch.sum(emb_triplet)).item()

        return score

    def score_table_s_r(self, s, r):
        rel = ('head', r, 'tail')

        s_index = self.entity2index[s]
        r_index = self.relation2index[rel]

        o_index_list = [i for i in range(len(self.entity2index))]

        score_list = [0 for i in range(len(self.entity2index))]

        for i, o_index in enumerate(o_index_list):
            score_list[i] = self.predict_score_with_s_r_o(s_index, r_index, o_index)

        s_list = [s for i in range(len(o_index_list))]
        r_list = [r for i in range(len(o_index_list))]

        o_list = []
        for o_index in o_index_list:
            o_list.append(self.index2entity[o_index])

        score_table = np.array([s_list, r_list, o_list, score_list]).T
        score_table = pd.DataFrame(score_table)

        score_table.columns = ['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI', 'SCORE']

        return score_table

    def score_table_r_o(self, r, o):
        rel = ('head', r, 'tail')

        r_index = self.relation2index[rel]
        o_index = self.entity2index[o]

        s_index_list = [i for i in range(len(self.entity2index))]

        score_list = [0 for i in range(len(self.entity2index))]

        for i, s_index in enumerate(s_index_list):
            score_list[i] = self.predict_score_with_s_r_o(s_index, r_index, o_index)

        o_list = [o for i in range(len(s_index_list))]
        r_list = [r for i in range(len(s_index_list))]

        s_list = []
        for s_index in s_index_list:
            s_list.append(self.index2entity[s_index])

        score_table = np.array([s_list, r_list, o_list, score_list]).T
        score_table = pd.DataFrame(score_table)

        score_table.columns = ['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI', 'SCORE']

        return score_table

    def get_embedding(self, cui):
        cui_index = self.entity2index[cui]
        embed = self.total_embed[cui_index]
        return embed.to("cpu").detach().numpy()

    def get_all_embedding(self):
        cui_list = [i for i in range(len(self.entity2index))]

        all_embedings = []

        for i in cui_list:
            cui = self.index2entity[i]
            embed = self.get_embedding(cui)
            cui_embed = [cui, embed.tolist()]
            all_embedings.append(cui_embed)

        return all_embedings


def main(args):
    graph = args.graph
    triples = pd.read_csv(graph, sep='\t')
    triples.sort_values("PMID", ascending=True, inplace=True, ignore_index=True)
    triples = triples[triples.columns[:3]]

    knowledge_graph = KnowledgeGraph(triples)

    knowledge_graph.load_model()

    # Get the scoring tables with relation and objective
    for r in args.rels:
        r_table = knowledge_graph.score_table_r_o(r, args.o)
        r_table.to_csv(r + "_score.tsv", index=False, sep='\t')

    # Get the embedding of knowledge graph
    all_embed = knowledge_graph.get_all_embedding()
    embed_dic = {}

    for embed in all_embed:
        embed_dic[embed[0]] = embed[1]

    json_file_path = "nodes_embedding.json"
    with open(json_file_path, 'w') as f:
        json.dump(embed_dic, f)

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
        "--rels",
        dest="rels",
        default="['TREATS','PREVENTS']",
        help="Relation(s) of triples",
    )


    parser.add_argument(
        "--obj",
        dest="o",
        default="C0002395",
        help="Object of triples",
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
        "--model_state_file",
        dest="model_state_file",
        default="model_state.pth",
    )

    args = parser.parse_args()
    args.rels = eval(args.rels)
    for r in args.rels:
        print(r)

    print(args.o)

    main(args)