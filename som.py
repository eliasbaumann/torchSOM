import torch
import logging

torch._logging.set_logs(dynamo=logging.WARNING, graph_breaks=True)
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.config.verbose = True


@torch.compile
def eucl_dist(d: torch.tensor, n: torch.tensor):
    # only implemented euclidean distance to compare against C-implemented SOM
    return ((d - n) ** 2).sum(1).sqrt()


@torch.compile
class torchSOM(torch.nn.Module):
    """
    create SOM module
    use step(...) to train for x iterations

    Parameters
    ----------
    n : int
        size of data (axis 0)
    rlen : int
        Number of times to loop over the training data for each MST
    threshold : torch.tensor
        radius cutoff
    thresholdStep : float
        change in radius per step
    alpha_start, alpha_end : float
        start and end learning rate
    """

    def __init__(
        self,
        n: int,
        niter: int,
        threshold: torch.tensor,
        thresholdStep: float,
        alpha_start: float,
        alpha_end: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.change = 1.0
        self.n = n
        self.niter = niter
        self.threshold = threshold
        self.thresholdStep = thresholdStep
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.rand_indices = torch.randint(0, n, size=[niter])
        self.threshold_cutoff = torch.tensor(0.5)

    def step(
        self, data: torch.tensor, nodes: torch.tensor, nhbrdist: torch.tensor, k: int
    ):
        """
        Parameters
        ----------
        data : torch.tensor
            input data
        nodes : torch.tensor
            updated SOM nodes
        nhbrdist : torch.tensor
        k : int
            current step
        """
        i = self.rand_indices[k]
        dist = eucl_dist(data[i], nodes)
        nearest = torch.argmin(dist)
        self.threshold = torch.max(self.threshold, self.threshold_cutoff)
        alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (
            k / self.niter
        )
        to_update = torch.index_select(nhbrdist, 1, nearest).flatten()
        to_update = to_update <= self.threshold
        tmp = data[i] - nodes[to_update]
        nodes[to_update] += tmp * alpha
        self.change += torch.abs(tmp).sum()
        self.threshold += self.thresholdStep
        return nodes


def low_level_som(
    data,
    nodes,
    nhbrdist,
    alpha_start: float,
    alpha_end: float,
    radius_start: float,
    radius_end: float,
    n: int,
    rlen: int,
):

    niter = rlen * n
    threshold = torch.tensor(radius_start)
    thresholdStep = (radius_start - radius_end) / niter
    som_module = torchSOM(n, niter, threshold, thresholdStep, alpha_start, alpha_end)
    k = 0

    while k < niter:
        if (k % n) == 0:
            if som_module.change < 1:
                k = niter
            som_module.change = 0.0
        nodes = som_module.step(data, nodes, nhbrdist, k)
        k += 1
    return nodes


@torch.compile
def neighborhood_distance(xdim: int, ydim: int):
    # grid is basically list of all coordinates in neighborhood
    #
    # [19]: np.meshgrid(np.arange(1,3+1), np.arange(1,3+1))
    # Out[19]:
    # [array([[1, 2, 3],
    #         [1, 2, 3],
    #         [1, 2, 3]]),
    #  array([[1, 1, 1],
    #         [2, 2, 2],
    #         [3, 3, 3]])]
    grid = torch.cartesian_prod(torch.arange(1, xdim + 1), torch.arange(1, ydim + 1))[
        :, [1, 0]
    ].float()
    # setting p=inf is the same as chebyshev distance, or Maximal distance
    return torch.cdist(grid, grid, p=torch.inf)


def som(
    data,
    xdim=10,
    ydim=10,
    rlen=10,
    alpha_range=(0.05, 0.01),
    radius_range=None,
    distf=2,
    nodes=None,
    importance=None,
    seed=42,
):
    """
    Build a self-organizing map

    Parameters
    ----------
    data : np.Typing.NDArray[np.dtype("d")]
        2D array containing the training observations
        shape: (observation_count, parameter_count)
    xdim : int
        Width of the grid
    ydim : int
        Height of the grid
    rlen : int
        Number of times to loop over the training data for each MST
    alpha_range : Tuple[float, float]
        Start and end learning rate
    radius_range : Tuple[float, float]
        Start and end radius. If None, radius is set to a reasonable value
        value based on the grid size i.e. xdim and ydim
    distf: int
        Distance function to use.
        1 = manhattan
        2 = euclidean
        3 = chebyshev
        4 = cosine
    nodes : np.Typing.NDArray[np.dtype("d")]
        Cluster centers to start with.
        shape = (xdim * ydim, parameter_count)
        If None, nodes are initialized by randomly selecting observations
    importance : np.Typing.NDArray[np.dtype("d")]
        Scale parameters columns of input data an importance weight
        shape = (parameter_count,)
    seed : int
        The random seed to use for node initialization, ignored if nodes is None.
        If None, runs with fully stochastic node initialization.

    Returns
    -------
    np.Typing.NDArray[]
    """
    nhbrdist = neighborhood_distance(xdim, ydim)

    if radius_range is None:
        # Let the radius have a sane default value based on the grid size
        radius_range = (torch.quantile(nhbrdist, q=0.67).item(), 0)

    n_nodes = xdim * ydim
    if nodes is None:
        # If we don't supply nodes, then initialize a seed and randomly sample xdim * ydim rows
        torch.manual_seed(seed)
        # torch.randint(low=0,high=data.shape[0],size=)
        nodes = data[torch.randperm(data.shape[0])[: xdim * ydim]]
        # nodes = data[np.random.choice(, xdim * ydim, replace=False), :]

    # if not data.flags['F_CONTIGUOUS']:
    #     data = np.asfortranarray(data)

    # assert data.dtype == np.dtype("d")
    # cdef double[::1,:] data_mv = data
    data_rows = data.shape[0]
    data_cols = data.shape[1]

    # if not nodes.flags['F_CONTIGUOUS']:
    #     nodes = np.asfortranarray(nodes)

    # assert nodes.dtype == np.dtype("d")
    # cdef double[::1,:] nodes_mv = nodes
    nodes_rows = nodes.shape[0]
    nodes_cols = nodes.shape[1]

    # if not nhbrdist.flags['F_CONTIGUOUS']:
    #     nhbrdist = np.asfortranarray(nhbrdist)

    # assert nhbrdist.dtype == np.dtype("d")
    # cdef double[::1,:] nhbrdist_mv = nhbrdist

    if nodes_cols != data_cols:
        raise Exception(
            f"When passing nodes, it must have the same number of columns as the data, nodes has {nodes_cols} columns, data has {data_cols} columns"
        )
    if nodes_rows != xdim * ydim:
        raise Exception(
            f"When passing nodes, it must have the same number of rows as xdim * ydim. nodes has {nodes_rows} rows, xdim * ydim = {xdim * ydim}"
        )

    if importance is not None:
        # scale the data by the importance weights
        raise NotImplementedError("importance weights not implemented yet")

    # xDists = torch.zeros((n_nodes), dtype=data.dtype)
    # cdef double [:] xDists_mv = xDists

    # if seed is not None:
    #     C_SEED_RAND(seed)
    # else:
    #     C_SEED_RAND(np.random.randint(low=1, high=65535))

    nodes = low_level_som(
        data,
        nodes,
        nhbrdist,
        alpha_range[0],
        alpha_range[1],
        radius_range[0],
        radius_range[1],
        data_rows,
        rlen,
    )
    # C_SOM(
    #     &data_mv[0, 0],
    #     &nodes_mv[0, 0],
    #     &nhbrdist_mv[0, 0],

    #     alpha_range[0],
    #     alpha_range[1],

    #     radius_range[0],
    #     radius_range[1],

    #     &xDists_mv[0],

    #     data_rows,
    #     data_cols,

    #     n_nodes,

    #     rlen,
    #     distf
    #     )

    return nodes


# def map_data_to_nodes(nodes, newdata, distf=2):
#     """Assign nearest node to each obersevation in newdata

#     Both nodes and newdata must represent the same parameters, in the same order.

#     Parameters
#     ----------
#     nodes : np.typing.NDArray[np.dtype("d")]
#         Nodes of the SOM.
#         shape = (node_count, parameter_count)
#         Fortan contiguous preffered
#     newdata: np.typing.NDArray[np.dtype("d")]
#         New observations to assign nodes.
#         shape = (observation_count, parameter_count)
#         Fortan contiguous preffered
#     distf: int
#         Distance function to use.
#         1 = manhattan
#         2 = euclidean
#         3 = chebyshev
#         4 = cosine

#     Returns
#     -------
#     (np.typing.NDArray[dtype("i")], np.typing.NDArray[np.dtype("d")])
#         The first array contains the node index assigned to each observation.
#             shape = (observation_count,)
#         The second array contains the distance to the node for each observation.
#             shape = (observation_count,)

#     """

#     if not nodes.flags['F_CONTIGUOUS']:
#         nodes = np.asfortranarray(nodes)
#     cdef double[::1,:] nodes_mv = nodes
#     nodes_rows = nodes.shape[0]
#     nodes_cols = nodes.shape[1]

#     if not newdata.flags['F_CONTIGUOUS']:
#         newdata = np.asfortranarray(newdata)
#     cdef double[::1,:] newdata_mv = newdata
#     newdata_rows = newdata.shape[0]
#     newdata_cols = newdata.shape[1]

#     nnClusters = np.zeros(newdata_rows, dtype=np.dtype("i"))
#     nnDists = np.zeros(newdata_rows, dtype=np.dtype("d"))
#     cdef int [:] nnClusters_mv = nnClusters
#     cdef double [:] nnDists_mv = nnDists

#     C_mapDataToNodes(
#         &newdata_mv[0, 0],
#         &nodes_mv[0, 0],
#         nodes_rows,
#         newdata_rows,
#         nodes_cols,
#         &nnClusters_mv[0],
#         &nnDists_mv[0],
#         distf
#         )

#     return (nnClusters, nnDists)


# from pyFlowSOM import map_data_to_nodes as pfsmap_data_to_nodes, som as pfssom
# from som import som


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    torch.manual_seed(1729)
    # generate example input data, rows are observations (e.g. cells), columns are features (e.g. proteins)
    df = pd.DataFrame(np.random.rand(500, 16))

    # alternatively, specify path to your own input data
    # df = pd.read_csv('path/to/som/input.csv')

    example_som_input_arr = torch.from_numpy(df.to_numpy())
    node_output2 = som(
        example_som_input_arr, xdim=10, ydim=10, rlen=10, seed=1, distf="eucl"
    )
