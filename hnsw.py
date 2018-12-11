# -*- coding: utf-8 -*-

from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random


class HNSW(object):
    """Hierarchical Navigable Small World (HNSW) data structure.

    Based on the work by Yury Malkov and Dmitry Yashunin, available at
    http://arxiv.org/pdf/1603.09320v2.pdf

    HNSWs allow performing approximate nearest neighbor search with
    arbitrary data and non-metric dissimilarity functions.
    """

    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def __init__(self, d, m=5, ef=200, m0=None, level_mult=None,
                 heuristic=True, vectorized=False):
        """d the dissimilarity function

        If vectorized is true, d can be called on lists as second argument
        to compare multiple elements with the first.

        See other parameters in http://arxiv.org/pdf/1603.09320v2.pdf"""

        self.data = []

        if vectorized:
            def d1(x, y):
                return d(x, [y])[0]

            self.distance = d1
            self.vectorized_distance = d
        else:
            self.distance = d

            def vd(x, ys):
                return [d(x, y) for y in ys]

            self.vectorized_distance = vd

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m) if level_mult is None else level_mult
        self._graphs = []
        self._enter_point = None

        self._select = (self._select_heuristic if heuristic
                        else self._select_naive)

    def add(self, elem, ef=None):
        """Add elem to the data structure"""

        if ef is None:
            ef = self._ef

        d = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # level at which the element will be inserted
        level = int(-log2(random()) * self._level_mult) + 1

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = d(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for g in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, g)
            # at these levels we have to insert elem; ep is a heap of
            # entry points.
            ep = [(-dist, point)]
            g0 = graphs[0]
            for g in reversed(graphs[:level]):
                level_m = m if g is not g0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, g, ef)
                # insert in g[idx] the best neighbors
                g[idx] = g_idx = {}
                self._select(g_idx, ep, level_m, g, heap=True)
                # assert len(g_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in g_idx.items():
                    self._select(g[j], (idx, dist), level_m, g)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx

    def balanced_add(self, elem, ef=None):
        """Add elem to the data structure.

        This is implemented in a different way with respect to the
        approach described by Malkov and Yashunin, resulting in a
        better balanced data structure.

        Rather than choosing randomly the level of an element, an
        element is raised to a higher level if its degree is m and
        no neighbor is in the higher level.
        """

        if ef is None:
            ef = self._ef

        d = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist = d(elem, data[point])
            pd = [(point, dist)]
            # navigate from the top to the bottom looking for the closest
            # node in each graph, and save in pd the closest found
            for g in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, g)
                pd.append((point, dist))
            # now go from bottom to the last level in which we should
            # add the new node
            for level, g in enumerate(graphs):
                level_m = m0 if level == 0 else m
                # find the candidate neighbors and select which ones to insert
                candidates = self._search_graph(elem, [(-dist, point)], g, ef)
                g[idx] = g_idx = {}
                self._select(g_idx, candidates, level_m, g, heap=True)
                # add reverse edges
                for j, dist in g_idx.items():
                    self._select(g[j], [idx, dist], level_m, g)
                    assert len(g[j]) <= level_m
                # stop here if the node has less than level_m neighbors
                if len(g_idx) < level_m:
                    return
                # or if at least one of them is in the upper level
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in g_idx):
                        return
                point, dist = pd.pop()
        graphs.append({idx: {}})
        self._enter_point = idx

    def search(self, q, k=None, ef=None):
        """Find the k points closest to q."""

        d = self.distance
        graphs = self._graphs
        point = self._enter_point

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = d(q, self.data[point])
        # look for the closest neighbor from the top to the 2nd level
        for g in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, g)
        # look for ef neighbors in the bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, g):
        """Equivalent to _search_graph when ef=1."""

        vd = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in g[c] if e not in visited]
            visited.update(edges)
            dists = vd(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
                    # break

        return best, best_dist

    def _search_graph(self, q, ep, g, ef):

        vd = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break

            edges = [e for e in g[c] if e not in visited]
            visited.update(edges)
            dists = vd(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_naive(self, d, to_insert, m, g, heap=False):

        if not heap:  # shortcut when we've got only one thing to insert
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        # so we have more than one item to insert, it's a bit more tricky
        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)  # smallest m distances
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):

        nb_dicts = [g[idx] for idx in d]

        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist)
                                      for mdist, idx in to_insert))

        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist)
                                              for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m

    def __getitem__(self, idx):
        """Returns a list of known neighbors of node at index idx."""

        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return


if __name__ == "__main__":
    dim = 200
    num_elements = 100

    import numpy as np

    # Generating sample data
    data = np.float32(np.random.random((num_elements, dim)))
    data_labels = np.arange(num_elements)

    from numpy.linalg import norm


    def d(a, b):
        return norm(a - b)


    hnsw = HNSW(d, m0=48, ef=200)

    hnsw.add(data)
    idx = hnsw.search(np.float32(np.random.random((1, dim))), 10, ef=200)
    print(idx)
