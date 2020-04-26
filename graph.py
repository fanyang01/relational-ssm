from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import scipy.sparse


def remove_self_loops(edges):
    indices = np.where(np.not_equal(edges[:, 0], edges[:, 1]))[0]
    if np.size(indices) > 0:
        return edges[indices]
    return np.zeros([0, 2], dtype=np.int32)


def lexsort(edges):
    sender, receiver = edges[:, 0], edges[:, 1]
    indices = np.lexsort((receiver, sender))
    return edges[indices], indices


def relabel(edges, centers):
    flatten = np.reshape(edges, -1)
    vertices, indices = np.unique(flatten, return_inverse=True)
    labels = np.arange(np.size(vertices), dtype=np.int32)
    flatten = labels[indices]
    relabeled_edges = np.reshape(flatten, np.shape(edges))
    relabeled_centers = np.searchsorted(vertices, centers)
    assert np.array_equal(centers, vertices[relabeled_centers])
    mask = np.zeros(np.size(vertices))
    mask[relabeled_centers] = 1
    return relabeled_edges, vertices, mask


class Sampler:
    def __init__(self, edges, num_vertices=None):
        '''Build the adjacency list representation of a graph.

        Args:
          edges: a lexicographical-ordered numpy array of shape (E, 2).
            Every vertex must have a self-loop edge.
          num_vertices: the number of vertices in the graph. If None, it
            will be `edges[-1, 0] + 1`.
        '''
        if num_vertices is None:
            num_vertices = (edges[-1, 0] + 1)
        assert num_vertices == (edges[-1, 0] + 1)

        offsets = np.full(num_vertices, -1, dtype=np.int32)
        lengths = np.full(num_vertices, 0, dtype=np.int32)
        neighbors = np.copy(edges[:, 1])
        prev, pos, length = -1, -1, 0
        for e in edges:
            pos += 1
            vertex = e[0]
            if vertex < prev:
                raise ValueError("Edge list is not lexicographical-ordered.")
            elif vertex == prev:
                length += 1
                continue
            if prev >= 0:
                lengths[prev] = length
            offsets[vertex] = pos
            prev, length = vertex, 1
        if prev >= 0:
            lengths[prev] = length

        adjacency_matrix = scipy.sparse.coo_matrix(
            (
                np.ones(np.shape(edges)[0], dtype=np.int8),
                (edges[:, 0], edges[:, 1])
            ), shape=(num_vertices, num_vertices)
        )
        assert adjacency_matrix.nnz == np.shape(edges)[0]
        assert np.prod(adjacency_matrix.diagonal()) == 1

        self._num_vertices = num_vertices
        self._neighbors = neighbors
        self._offsets = offsets
        self._lengths = lengths
        self._adjacency_matrix = adjacency_matrix.tocsr()

    def adjacent_vertices(self, vertex):
        assert vertex >= 0
        length = self._lengths[vertex]
        if np.equal(length, 0):
            raise ValueError("Every vertex SHOULD have at least one neighbor.")
        start = self._offsets[vertex]
        end = start + self._lengths[vertex]
        neighbors = self._neighbors[start:end]
        return neighbors, length

    def random_walk_with_restart(self, length,
                                 restart_prob=0.15, win_size=2,
                                 init_vertex=None, seed=None):
        '''
        Self-loop edges are ignored during random walking. The walk will
        terminate immediately after reaching some vertex of out-degree 0.
        '''
        if init_vertex is None:
            init_vertex = np.random.randint(0, self._num_vertices)
        rand_ints = np.random.randint(0, self._num_vertices, size=length)
        rand_restarts = np.random.rand(length)

        current, path, paths = init_vertex, [], []
        for t in range(length):
            path.append(current)
            neighbors, N = self.adjacent_vertices(current)
            if N == 0 or (N == 1 and neighbors[0] == current):
                break
            if current != init_vertex and rand_restarts[t] < restart_prob:
                paths.append(path)
                current, path = init_vertex, []
                continue
            idx = rand_ints[t] % N
            next_vertex = neighbors[idx]
            if next_vertex == current:
                next_vertex = neighbors[(idx + 1) % N]
            current = next_vertex
        if len(path) > 0:
            paths.append(path)

        return paths

    def random_walk(self, length, seed=None):
        paths = self.random_walk_with_restart(length, restart_prob=0.0)
        assert len(paths) == 1
        return paths[0]

    def random_walk_for_fixed_len(self, length, seed=None):
        paths, len_acc = [], 0
        while len_acc < length:
            new_path = self.random_walk(length - len_acc)
            len_acc += len(new_path)
            paths.append(new_path)
        return paths

    def approx_unigram_dist(self, power=0.75):
        paths = self.random_walk_for_fixed_len(self._num_vertices * 10)
        samples = np.concatenate(paths, axis=0)
        frequency = np.bincount(samples, minlength=self._num_vertices)
        frequency = np.add(frequency, 1)
        raised = np.power(frequency, power)
        return np.divide(raised, np.sum(raised))

    def negative_sampling(
            self, vertices, num_samples_per_vertex, unigram_dist):
        num_vertices = np.size(vertices)
        num_total = num_vertices * num_samples_per_vertex
        samples = np.random.choice(
            self._num_vertices, size=num_total, p=unigram_dist)
        vertices = np.tile(vertices, num_samples_per_vertex)

        def filter(adj_matrix, senders, receivers):
            entries = adj_matrix[senders, receivers]
            entries = np.array(entries)[0]
            mask = np.equal(entries, 0)
            non_edges_from = senders[mask]
            non_edges_to = receivers[mask]
            non_edges = np.stack([non_edges_from, non_edges_to], axis=1)
            return non_edges

        non_edges_0 = filter(self._adjacency_matrix, vertices, samples)
        non_edges_1 = filter(self._adjacency_matrix, samples, vertices)
        non_edges = np.concatenate([non_edges_0, non_edges_1])
        return non_edges

    def random_walk_centered(self, length,
                             restart_prob=0.15, win_size=2,
                             init_vertex=None, seed=None):
        paths, len_acc = [], 0
        while len_acc < length:
            new_paths = self.random_walk_with_restart(
                length - len_acc,
                restart_prob=restart_prob, win_size=win_size,
                init_vertex=init_vertex
            )
            if init_vertex is None:
                init_vertex = new_paths[0][0]
            for path in new_paths:
                len_acc += len(path)
            paths.extend(new_paths)
        return paths

    def gen_skip_gram(self, paths, win_size):
        '''
        Self-loop edges are implicitly added for all vertices encountered
        by the random walk.
        '''
        centers = []
        senders, receivers = [], []
        len_total = 0

        for path in paths:
            centers.append(path[0])
            len_total += len(path)
            for i in range(len(path)):
                senders.append(path[i])
                receivers.append(path[i])
                for j in range(win_size - 1):
                    if i + j + 1 < len(path):
                        senders.append(path[i])
                        receivers.append(path[i + j + 1])

        values = np.ones(len(senders), dtype=np.int8)
        graph = scipy.sparse.coo_matrix(
            (values, (senders, receivers))
        ).tocsr()
        return np.transpose(graph.nonzero()), np.unique(centers)

    def skip_gram(self, rand_walk_len, win_size,
                  restart_prob=0.15, init_vertex=None):
        assert rand_walk_len > 0
        paths = self.random_walk_centered(
            length=rand_walk_len, win_size=win_size,
            restart_prob=restart_prob, init_vertex=init_vertex
        )
        return self.gen_skip_gram(paths, win_size)

    def filter_by_negative_sampling(self, edges,
                                    num_neg_samples,
                                    unigram_distribution):
        vertices = np.unique(np.reshape(edges, -1))
        non_edges = self.negative_sampling(
            vertices=vertices,
            num_samples_per_vertex=num_neg_samples,
            unigram_dist=unigram_distribution
        )
        shape = (self._num_vertices, self._num_vertices)
        pos_values = np.ones(np.shape(edges)[0], dtype=np.int8)
        pos_graph = scipy.sparse.coo_matrix(
            (pos_values, (edges[:, 0], edges[:, 1])),
            shape=shape
        ).tocsr()
        pos_graph = scipy.sparse.coo_matrix(
            (
                np.full(pos_graph.nnz, 1, dtype=np.int8),
                pos_graph.nonzero()
            ),
            shape=shape
        ).tocsr()
        neg_values = np.ones(np.shape(non_edges)[0], dtype=np.int8)
        neg_graph = scipy.sparse.coo_matrix(
            (neg_values, (non_edges[:, 0], non_edges[:, 1])),
            shape=shape
        ).tocsr()
        neg_graph = scipy.sparse.coo_matrix(
            (
                np.full(neg_graph.nnz, -1, dtype=np.int8),
                neg_graph.nonzero()
            ),
            shape=shape
        ).tocsr()

        graph = neg_graph.multiply(pos_graph) + pos_graph
        return np.transpose(graph.nonzero())

    def skip_gram_with_neg_sampling(self,
                                    rand_walk_len, win_size,
                                    num_neg_samples, unigram_distribution,
                                    restart_prob=0.15, init_vertex=None):
        edges, centers = self.skip_gram(
            rand_walk_len, win_size,
            restart_prob=restart_prob, init_vertex=init_vertex
        )
        filtered_edges = self.filter_by_negative_sampling(
            edges, num_neg_samples=num_neg_samples,
            unigram_distribution=unigram_distribution
        )
        return filtered_edges, centers

    def sample_with_rand_walk(self,
                              min_num_vertices,
                              rand_walk_len,
                              win_size,
                              num_neg_samples,
                              unigram_distribution,
                              restart_prob=0.15):
        t, num_vertices = 0, 0
        edges, centers = None, None
        while num_vertices < min_num_vertices and t < 100:
            t += 1
            new_edges, new_centers = self.skip_gram_with_neg_sampling(
                rand_walk_len=rand_walk_len,
                restart_prob=restart_prob,
                win_size=win_size,
                num_neg_samples=num_neg_samples,
                unigram_distribution=unigram_distribution
            )
            if edges is None:
                edges, centers = new_edges, new_centers
            else:
                edges = np.concatenate([edges, new_edges], axis=0)
                centers = np.concatenate([centers, new_centers], axis=0)
            num_vertices = np.size(np.unique(np.reshape(edges, -1)))
        if t >= 100:
            raise ValueError("Possible BUG: Restart for too many times!")

        values = np.ones(np.shape(edges)[0], dtype=np.int8)
        graph = scipy.sparse.coo_matrix(
            (values, (edges[:, 0], edges[:, 1]))
        ).tocsr()
        return np.transpose(graph.nonzero()), np.unique(centers)


class BiSampler:
    def __init__(self, edges, num_vertices=None):
        reverse_edges, _ = lexsort(np.fliplr(edges))
        self._sampler = Sampler(edges, num_vertices)
        self._r_sampler = Sampler(reverse_edges, num_vertices)

    def approx_unigram_distribution(self, power=0.75):
        return self._sampler.approx_unigram_dist(power=power)

    def sample_with_bi_rand_walk(self,
                                 min_num_vertices,
                                 rand_walk_len,
                                 win_size,
                                 num_neg_samples,
                                 unigram_distribution,
                                 restart_prob=0.15):
        rand_walk_config = dict(
            rand_walk_len=rand_walk_len,
            win_size=win_size,
            restart_prob=restart_prob
        )
        t, num_vertices = 0, 0
        edges, centers = None, None
        while num_vertices < min_num_vertices and t < 100:
            t += 1
            sampled_r_edges, sampled_r_centers = self._r_sampler.skip_gram(
                **rand_walk_config
            )
            assert len(sampled_r_centers) == 1

            sampled_edges, sampled_centers = self._sampler.skip_gram(
                **rand_walk_config, init_vertex=sampled_r_centers[0]
            )
            assert np.array_equal(sampled_r_centers, sampled_centers)

            merged_edges = np.concatenate(
                [np.fliplr(sampled_r_edges), sampled_edges], axis=0
            )
            new_edges = self._sampler.filter_by_negative_sampling(
                merged_edges, num_neg_samples=num_neg_samples,
                unigram_distribution=unigram_distribution
            )
            new_centers = sampled_centers

            if edges is None:
                edges, centers = new_edges, new_centers
            else:
                edges = np.concatenate([edges, new_edges], axis=0)
                centers = np.concatenate([centers, new_centers], axis=0)
            num_vertices = np.size(np.unique(np.reshape(edges, -1)))
        if t >= 100:
            raise ValueError("Possible BUG: Restart for too many times!")

        values = np.ones(np.shape(edges)[0], dtype=np.int8)
        graph = scipy.sparse.coo_matrix(
            (values, (edges[:, 0], edges[:, 1]))
        ).tocsr()
        return np.transpose(graph.nonzero()), np.unique(centers)
