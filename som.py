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
        threshold_step: float,
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
        self.threshold_step = threshold_step
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

        Returns
        -------
        torch.tensor
            updated SOM nodes
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
        self.threshold += self.threshold_step
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
    """
    Calculate the self-organizing map

    Parameters
    ----------
    data : torch.tensor
        input data
    nodes : torch.tensor
        updated SOM nodes
    nhbrdist : torch.tensor
    alpha_start, alpha_end : float
        start and end learning rate
    radius_start, radius_end : float
        start and end radius
    n : int
        size of data (axis 0)
    rlen : int
        Number of times to loop over the training data for each MST

    Returns
    -------
    torch.tensor
        updated SOM nodes
    """

    niter = rlen * n
    threshold = torch.tensor(radius_start)
    threshold_step = (radius_start - radius_end) / niter
    som_module = torchSOM(n, niter, threshold, threshold_step, alpha_start, alpha_end)
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
    """
    Calculate the distance between each node in the grid
    """

    grid = torch.cartesian_prod(torch.arange(1, xdim + 1), torch.arange(1, ydim + 1))[
        :, [1, 0]
    ].float()

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

    # n_nodes = xdim * ydim
    if nodes is None:
        # If we don't supply nodes, then initialize a seed and randomly sample xdim * ydim rows
        torch.manual_seed(seed)
        nodes = data[torch.randperm(data.shape[0])[: xdim * ydim]]

    data_rows = data.shape[0]
    data_cols = data.shape[1]

    nodes_rows = nodes.shape[0]
    nodes_cols = nodes.shape[1]
    assert (
        data_cols != nodes_cols
    ), f"When passing nodes, it must have the same number of columns as the data, nodes has {nodes_cols} columns, data has {data_cols} columns"
    assert (
        data_rows != xdim * ydim
    ), f"When passing nodes, it must have the same number of rows as xdim * ydim. nodes has {nodes_rows} rows, xdim * ydim = {xdim * ydim}"

    if importance is not None:
        # scale the data by the importance weights
        raise NotImplementedError("importance weights not implemented yet")

    # Distance functions are currently not implemented except for euclidean
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

    return nodes


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
