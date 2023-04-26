import numpy as np
from sklearn.model_selection import train_test_split


class Data():
    def __init__(self, graph):
        self.df_graph = graph.copy()
        self.generate_dictionary()
        self.total_data = self.generate_dataset()
        self.num_nodes, self.num_rels, self.num_edges = self.get_stats()
        self.get_train_test_data(0.1, 0.1)

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

        return num_nodes, num_rels, num_edges

    def get_train_test_data(self, valid_p, test_p):
        # Time Silcing applied
        train_data, valid_data = train_test_split(self.total_data, test_size=(valid_p + test_p), shuffle=False)
        valid_data, test_data = train_test_split(valid_data, test_size=(valid_p / (valid_p + test_p)), shuffle=False)

        return train_data, valid_data, test_data
