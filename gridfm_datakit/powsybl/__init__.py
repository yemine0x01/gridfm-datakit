"""
PowSybl module for gridfm_datakit.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

from gridfm_datakit.network import Network, load_net_from_file

from .api import check_powsybl_available, pypowsybl
from .convert import from_powsybl, to_powsybl, ConversionOptions


def _check_scipy_available() -> None:
    """Check if scipy is available for .mat file operations."""
    try:
        import scipy.io  # noqa: F401
    except ImportError:
        raise ImportError(
            "scipy is required for .mat file operations. "
            "Install it with: pip install scipy"
        )


def to_mat_file(network: Network, output_path: Union[str, Path]) -> Path:
    """
    Convert a gridfm_datakit Network to a MATPOWER .mat binary file.

    This creates a .mat file that can be loaded by pypowsybl's MATPOWER importer.
    The .mat file contains an 'mpc' struct with MATPOWER case data.

    Parameters
    ----------
    network : Network
        The gridfm_datakit Network object to convert.
    output_path : str or Path
        Path where the .mat file will be saved.

    Returns
    -------
    Path
        The path to the created .mat file.

    Example
    -------
    >>> from gridfm_datakit.network import load_net_from_pglib
    >>> from gridfm_datakit.powsybl import to_mat_file
    >>>
    >>> net = load_net_from_pglib("case14_ieee")
    >>> mat_path = to_mat_file(net, "case14.mat")
    >>> print(f"Created {mat_path}")
    """
    _check_scipy_available()
    import scipy.io

    output_path = Path(output_path)

    # Build MATPOWER mpc struct
    # MATPOWER expects specific matrix formats
    mpc = {
        "version": "2",
        "baseMVA": np.array([[network.baseMVA]]),
        "bus": network.buses,
        "gen": network.gens,
        "branch": network.branches,
    }

    # Add gencost if available
    if hasattr(network, "gencosts") and network.gencosts is not None:
        mpc["gencost"] = network.gencosts

    # Save as .mat file with 'mpc' struct
    scipy.io.savemat(str(output_path), {"mpc": mpc})

    return output_path


def convert_m_to_mat(m_file_path: Union[str, Path], output_path: Union[str, Path] = None) -> Path:
    """
    Convert a MATPOWER .m text file to a .mat binary file.

    This is useful because pypowsybl can load .mat files but not .m files.

    Parameters
    ----------
    m_file_path : str or Path
        Path to the MATPOWER .m file.
    output_path : str or Path, optional
        Path where the .mat file will be saved. If not provided,
        uses the same path as the input with .mat extension.

    Returns
    -------
    Path
        The path to the created .mat file.

    Example
    -------
    >>> from gridfm_datakit.powsybl import convert_m_to_mat
    >>>
    >>> mat_path = convert_m_to_mat("case14.m")
    >>> print(f"Converted to {mat_path}")
    """
    m_file_path = Path(m_file_path)

    if not m_file_path.exists():
        raise FileNotFoundError(f"File not found: {m_file_path}")

    if output_path is None:
        output_path = m_file_path.with_suffix(".mat")
    else:
        output_path = Path(output_path)

    # Load the .m file using gridfm_datakit
    network = load_net_from_file(str(m_file_path))

    # Convert to .mat file
    return to_mat_file(network, output_path)


@dataclass
class NetworkMetadata:
    """
    Metadata extracted from network files that is not part of the PowSybl model.

    Attributes:
        gen_costs: Polynomial cost coefficients (c2, c1, c0) for each generator.
                   Keys are generator IDs, values are tuples of coefficients.
        extra: Additional metadata that may be loaded from the file.
    """

    gen_costs: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedNetwork:
    """
    Container for a loaded network with all associated data.

    Attributes:
        pp_net: The PowSybl network object.
        gfm_net: The gridfm_datakit Network object.
        metadata: Metadata extracted from the network file (e.g., gen_costs).
    """

    pp_net: Any  # pypowsybl.network.Network
    gfm_net: Network
    metadata: NetworkMetadata


def load_metadata(network_path: str) -> NetworkMetadata:
    """
    Load metadata (e.g., generator costs) from a network file.

    This function extracts metadata that is not directly supported by PowSybl
    but may be present in MATPOWER or other file formats.

    Parameters
    ----------
    network_path : str
        Path to the network file.

    Returns
    -------
    NetworkMetadata
        Metadata extracted from the file, including generator costs if available.
    """
    path = Path(network_path)
    metadata = NetworkMetadata()

    # For MATPOWER .m files, try to extract gen_costs
    if path.suffix.lower() == ".m":
        metadata.gen_costs = _parse_matpower_gencost(path)

    return metadata


def _parse_matpower_gencost(file_path: Path) -> Dict[str, Tuple[float, ...]]:
    """
    Parse generator cost data from a MATPOWER .m file.

    Parameters
    ----------
    file_path : Path
        Path to the MATPOWER file.

    Returns
    -------
    Dict[str, Tuple[float, ...]]
        Dictionary mapping generator index (as string) to cost coefficients.
        For polynomial costs, returns (c2, c1, c0) for quadratic or (c1, c0) for linear.
    """
    gen_costs: Dict[str, Tuple[float, ...]] = {}

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Find the gencost section
        import re

        # Pattern to match mpc.gencost = [ ... ];
        gencost_pattern = r"mpc\.gencost\s*=\s*\[(.*?)\];"
        match = re.search(gencost_pattern, content, re.DOTALL)

        if match:
            gencost_data = match.group(1).strip()
            lines = [
                line.strip()
                for line in gencost_data.split("\n")
                if line.strip() and not line.strip().startswith("%")
            ]

            for idx, line in enumerate(lines):
                # Remove MATLAB comments (everything after %)
                if "%" in line:
                    line = line.split("%")[0]

                # Remove all semicolons (row delimiters in MATLAB)
                line = line.replace(";", "")

                # Strip whitespace
                line = line.strip()
                if not line:
                    continue

                # Parse the numeric values
                try:
                    values = [float(v) for v in line.split()]
                except ValueError:
                    continue

                if len(values) >= 5:
                    # MATPOWER gencost format:
                    # MODEL, STARTUP, SHUTDOWN, NCOST, COST...
                    # For polynomial (MODEL=2): NCOST coefficients follow
                    model = int(values[0])
                    ncost = int(values[3])

                    if model == 2 and len(values) >= 4 + ncost:
                        # Polynomial cost: extract coefficients
                        coeffs = tuple(values[4 : 4 + ncost])
                        gen_costs[str(idx)] = coeffs

    except Exception:
        # If parsing fails, return empty dict (will use defaults)
        pass

    return gen_costs


def load_net(network_path: str) -> LoadedNetwork:
    """
    Load a network from a file using PowSybl.

    Supports all pypowsybl formats (XIIDM, CGMES, IEEE-CDF, PSS/E, UCTE, etc.)
    plus MATPOWER .m text files (automatically converted to .mat format).

    Parameters
    ----------
    network_path : str
        Path to the file. For MATPOWER .m files, the file is automatically
        converted to .mat format before loading with pypowsybl.

    Returns
    -------
    LoadedNetwork
        A LoadedNetwork object containing:
        - pb_net: The PowSybl network object
        - gfm: The gridfm_datakit Network object
        - metadata: Metadata including gen_costs from the file
    """
    check_powsybl_available()

    path = Path(network_path)
    if not path.is_file():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    # Load metadata (including gen_costs) from the file
    metadata = load_metadata(network_path)

    # Handle MATPOWER .m files by converting to .mat first
    if path.suffix.lower() == ".m":
        # pypowsybl cannot load .m text files directly, but can load .mat binary files
        # Load the .m file using gridfm_datakit and convert to temp .mat file
        gfm_net = load_net_from_file(str(path))

        # Create a temporary .mat file
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            mat_path = Path(tmp.name)

        try:
            to_mat_file(gfm_net, mat_path)
            pp_net = pypowsybl.network.load(str(mat_path))
        finally:
            # Clean up temp file
            mat_path.unlink(missing_ok=True)

        # Extract gen_costs from the loaded network
        gen_costs = _extract_gen_costs_from_network(gfm_net)
        metadata = NetworkMetadata(gen_costs=gen_costs)
    else:
        # For other formats (XIIDM, CGMES, etc.), load directly with pypowsybl
        pp_net = pypowsybl.network.load(network_path)

        # Create conversion options with gen_costs from metadata
        options = ConversionOptions(gen_costs=metadata.gen_costs)

        # Convert to gridfm_datakit Network
        gfm_net = from_powsybl(pp_net, options=options)

    return LoadedNetwork(pp_net=pp_net, gfm_net=gfm_net, metadata=metadata)


def _extract_gen_costs_from_network(network: Network) -> Dict[str, Tuple[float, ...]]:
    """
    Extract generator cost coefficients from a gridfm_datakit Network.

    Parameters
    ----------
    network : Network
        The gridfm_datakit Network object.

    Returns
    -------
    Dict[str, Tuple[float, ...]]
        Dictionary mapping generator index (as string) to cost coefficients.
    """
    gen_costs: Dict[str, Tuple[float, ...]] = {}

    if not hasattr(network, "gencosts") or network.gencosts is None:
        return gen_costs

    gencost_matrix = network.gencosts

    # MATPOWER gencost format:
    # Col 0 (MODEL): Cost model type (1=piecewise linear, 2=polynomial)
    # Col 1 (STARTUP): Startup cost
    # Col 2 (SHUTDOWN): Shutdown cost
    # Col 3 (NCOST): Number of cost coefficients
    # Col 4+ (COST): Cost coefficients
    MODEL_IDX = 0
    NCOST_IDX = 3
    COST_IDX = 4
    POLYNOMIAL = 2

    for i in range(gencost_matrix.shape[0]):
        model = int(gencost_matrix[i, MODEL_IDX])
        ncost = int(gencost_matrix[i, NCOST_IDX])

        if model == POLYNOMIAL and ncost > 0:
            coeffs = tuple(gencost_matrix[i, COST_IDX : COST_IDX + ncost])
            gen_costs[str(i)] = coeffs

    return gen_costs


def convert_net(network: Network, network_id: str = "network") -> LoadedNetwork:
    """
    Convert a gridfm_datakit Network to a LoadedNetwork with pypowsybl representation.

    This function takes an existing gridfm_datakit Network and creates a LoadedNetwork
    containing the converted pypowsybl network, the original gfm network, and extracted
    metadata (including gen_costs).

    Parameters
    ----------
    network : Network
        The gridfm_datakit Network object to convert.
    network_id : str, optional
        ID for the created pypowsybl network (default: "network").

    Returns
    -------
    LoadedNetwork
        A LoadedNetwork object containing:
        - pp_net: The converted PowSybl network object
        - gfm_net: The original gridfm_datakit Network object
        - metadata: Metadata extracted from the network (including gen_costs)

    Example
    -------
    >>> from gridfm_datakit.network import load_net_from_pglib
    >>> from gridfm_datakit.powsybl import convert_net
    >>>
    >>> # Load a network using gridfm_datakit
    >>> gfm_net = load_net_from_pglib("case14_ieee")
    >>>
    >>> # Convert to LoadedNetwork with pypowsybl representation
    >>> loaded = convert_net(gfm_net)
    >>>
    >>> # Access the pypowsybl network
    >>> print(loaded.pp_net.get_buses())
    >>>
    >>> # Access the original gfm network
    >>> print(loaded.gfm_net.buses.shape)
    >>>
    >>> # Access the gen_costs
    >>> print(loaded.metadata.gen_costs)
    """
    check_powsybl_available()

    # Extract gen_costs from the gfm network's gencost matrix
    gen_costs = _extract_gen_costs_from_network(network)

    # Create metadata with extracted gen_costs
    metadata = NetworkMetadata(gen_costs=gen_costs)

    # Convert to pypowsybl network
    pb_net = to_powsybl(network, network_id=network_id)

    return LoadedNetwork(pp_net=pb_net, gfm_net=network, metadata=metadata)


__all__ = [
    "load_net",
    "load_metadata",
    "convert_net",
    "to_mat_file",
    "convert_m_to_mat",
    "from_powsybl",
    "to_powsybl",
    "ConversionOptions",
    "LoadedNetwork",
    "NetworkMetadata",
]
