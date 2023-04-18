import numpy as np
import torch
import dgl


def compute_degree_norm(g: dgl.DGLGraph):
    g = g.local_var()
    in_degs = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_degs
    norm[np.isinf(norm)] = 0
    return norm


def build_graph_from_triples(num_nodes, num_rels, triples):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triples
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))  # add reversed relations
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))  # [(dst, src, rel), (), ()...]
    dst, src, rel = np.array(edges).transpose()  # [rel_num*2]
    g.add_edges(src, dst)
    nodes_norm = compute_degree_norm(g)  # 1./in-degree
    return g, rel.astype('int64'), nodes_norm.astype('float32')


def build_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    return build_graph_from_triples(num_nodes, num_rels, triples=(src, rel, dst))


def node_norm_2_edge_norm(g: dgl.DGLGraph, node_norm):
    g = g.local_var()
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def get_adj(num_nodes, triplets):
    adj_list = [[] for _ in range(num_nodes)]
    for i, triple in enumerate(triplets):
        src, dst = triple[0], triple[2]
        # both directions have same id
        adj_list[src].append([i, dst])  # [edge_id, dst]
        adj_list[dst].append([i, src])

    adj_list = [np.array(n) for n in adj_list]
    return adj_list


def negative_sampling(pos_samples, num_entity, negative_rate):
    batch_size = len(pos_samples)
    generate_num = batch_size * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))  # [generate_num, 3]
    labels = np.zeros(batch_size * (negative_rate + 1), dtype=np.float32)
    labels[:batch_size] = 1
    values = np.random.randint(num_entity, size=generate_num)
    choices = np.random.uniform(size=generate_num)
    sub = choices > 0.5
    obj = choices <= 0.5
    # randomly replace sbj or obj
    neg_samples[sub, 0] = values[sub]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


def sample_edge_uniform(n_triplets, sample_size):
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    edges = np.zeros((sample_size,), dtype=np.int32)

    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            # all nodes are unseen, pick one node uniformly
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        p = weights / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=p)
        chosen_adj_list = adj_list[chosen_vertex]

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))  # chose one edge linked to chosen vertex
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            # this edge is already picked
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number  # pick this edge
        other_vertex = chosen_edge[1]  # another nodes on this edge
        picked[edge_number] = True  # this edge is picked
        sample_counts[chosen_vertex] -= 1  # in-degree of chosen-vertex minus one (this edge is deleted from graph
        sample_counts[other_vertex] -= 1
        seen[chosen_vertex] = True
        seen[other_vertex] = True

    return edges


def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_rels, adj_list, degrees, negative_rate,
                                      sampler='uniform'):
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(len(triplets), sample_size)  # edge id
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    # edges: used to re-number nodes
    uniq_v, edges = np.unique([src, dst], return_inverse=True)
    # relabel nodes to have consecutive node ids
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack([src, rel, dst]).transpose()  # [sample_size, 3]

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v), negative_rate)

    # further split graph, only part of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples

    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size)

    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    g, rel, norm = build_graph_from_triples(len(uniq_v), num_rels, (src, rel, dst))

    return g, uniq_v, rel, norm, samples, labels


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def perturb_and_get_raw_rank(emb, w, a, r, b, test_size, batch_size=100):
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    emb = emb.transpose(0, 1)
    w = w.transpose(0, 1)
    for idx in range(n_batch):
        batch_start = idx * batch_size
        batch_end = (idx + 1) * batch_size
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = emb[:,batch_a] * w[:,batch_r]
        emb_ar = emb_ar.unsqueeze(2)
        emb_c = emb.unsqueeze(1)

        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c)
        score = torch.sum(out_prod, dim=0).sigmoid()
        target = b[batch_start: batch_end]

        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
        ranks.append(indices[:, 1].view(-1))
    return torch.cat(ranks)


def filter(triplets_to_filter, target_s, target_r, target_o, num_nodes, neg_sample_size_eval, filter_o=True):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)

    # Add the ground truth node first
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]

    while len(candidate_nodes) < (neg_sample_size_eval + 1):
        e = np.random.randint(0, num_nodes)
        triplet = (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        # Do not consider a node if it leads to a real triplet
        if triplet not in triplets_to_filter and triplet not in candidate_nodes:
            candidate_nodes.append(e)
    return torch.LongTensor(candidate_nodes)


def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter, neg_sample_size_eval, filter_o=True):
    num_nodes = emb.shape[0]
    ranks = []
    for idx in range(test_size):
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(triplets_to_filter, target_s, target_r,
                                 target_o, num_nodes, neg_sample_size_eval, filter_o=filter_o)
        if filter_o:
            emb_s = emb[target_s]  # Vector
            emb_o = emb[candidate_nodes]  # A set of vectors
        else:
            emb_s = emb[candidate_nodes]
            emb_o = emb[target_o]
        target_idx = 0
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o  # Distmult
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))

        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def _calc_mrr(emb, w, test_triplets, total_data, batch_size, neg_sample_size_eval, hits, filter=False):
    with torch.no_grad():
        s, r, o = test_triplets[:,0], test_triplets[:,1], test_triplets[:,2]
        test_size = len(s)

        if filter:
            triplets_to_filter = {tuple(triplet) for triplet in total_data.tolist()}
            ranks_s = perturb_and_get_filtered_rank(emb, w, s, r, o, test_size,
                                                    triplets_to_filter, neg_sample_size_eval, filter_o=False)
            ranks_o = perturb_and_get_filtered_rank(emb, w, s, r, o,
                                                    test_size, triplets_to_filter, neg_sample_size_eval)
        else:
            ranks_s = perturb_and_get_raw_rank(emb, w, o, r, s, test_size, batch_size)
            ranks_o = perturb_and_get_raw_rank(emb, w, s, r, o, test_size, batch_size)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mr = torch.mean(ranks.float()).item()
        mrr = torch.mean(1.0 / ranks.float()).item()
        hits_dict = dict()
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            hits_dict[hit] = avg_count

    return mr, mrr, hits_dict


def calc_mrr(emb, w, test_triplets, total_data, batch_size=100, neg_sample_size_eval=20, hits=[1, 3, 10], eval_p="filtered"):
    if eval_p == "filtered":
        mr, mrr, hits_dict = _calc_mrr(emb, w, test_triplets, total_data, batch_size, neg_sample_size_eval, hits, filter=True)
    else:
        mr, mrr, hits_dict = _calc_mrr(emb, w, test_triplets, total_data, batch_size, hits)
    return mr, mrr, hits_dict