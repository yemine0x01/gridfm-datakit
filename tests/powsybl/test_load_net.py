"""
Test cases for gridfm_datakit.powsybl file loading functions.

Tests: load_net, load_metadata, to_mat_file, convert_m_to_mat
"""

import pytest
from pathlib import Path

from gridfm_datakit.powsybl.api import is_powsybl_available
from gridfm_datakit.network import Network

pytestmark = pytest.mark.skipif(
    is_powsybl_available() is False,
    reason="pypowsybl is not installed. Install with: pip install gridfm-datakit[powsybl]",
)


@pytest.fixture
def xiidm_case14_path(tmp_path):
    """Create a pypowsybl IEEE14 network and save to XIIDM format."""
    from gridfm_datakit.powsybl import pypowsybl as pp
    pp_net = pp.network.create_ieee14()
    xiidm_file = tmp_path / "ieee14.xiidm"
    pp_net.save(str(xiidm_file))
    return str(xiidm_file)


@pytest.fixture
def matpower_case14_path():
    """Get path to the MATPOWER case14 file."""
    grids_dir = Path(__file__).parent.parent.parent / "gridfm_datakit" / "grids"
    return str(grids_dir / "pglib_opf_case14_ieee.m")


class TestLoadNet:
    """Test cases for the load_net function."""

    def test_returns_loaded_network(self, xiidm_case14_path):
        """Test that load_net returns a LoadedNetwork with all components."""
        from gridfm_datakit.powsybl import load_net, LoadedNetwork, NetworkMetadata

        loaded = load_net(xiidm_case14_path)

        assert isinstance(loaded, LoadedNetwork)
        assert loaded.pp_net is not None
        assert hasattr(loaded.pp_net, "get_buses")
        assert isinstance(loaded.gfm_net, Network)
        assert isinstance(loaded.metadata, NetworkMetadata)

    def test_bus_count_matches(self, xiidm_case14_path):
        """Test that pypowsybl and gfm have matching bus counts."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)

        assert len(loaded.pp_net.get_buses()) == loaded.gfm_net.buses.shape[0] == 14

    def test_gen_and_branch_counts(self, xiidm_case14_path):
        """Test that generators and branches are loaded correctly."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)

        assert loaded.gfm_net.gens.shape[0] == 5
        assert loaded.gfm_net.branches.shape[0] == 20

    def test_gen_costs_empty_for_xiidm(self, xiidm_case14_path):
        """Test that gen_costs are empty for XIIDM files (no gencost data)."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(xiidm_case14_path)
        assert len(loaded.metadata.gen_costs) == 0

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        from gridfm_datakit.powsybl import load_net

        with pytest.raises(FileNotFoundError):
            load_net("/nonexistent/path/to/network.xiidm")


class TestLoadNetMatpower:
    """Test cases for load_net with MATPOWER .m files."""

    def test_returns_loaded_network(self, matpower_case14_path):
        """Test that load_net works with .m files."""
        from gridfm_datakit.powsybl import load_net, LoadedNetwork

        loaded = load_net(matpower_case14_path)

        assert isinstance(loaded, LoadedNetwork)
        assert loaded.pp_net is not None
        assert isinstance(loaded.gfm_net, Network)

    def test_extracts_gen_costs(self, matpower_case14_path):
        """Test that load_net with .m file extracts gen_costs."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(matpower_case14_path)
        assert len(loaded.metadata.gen_costs) > 0

    def test_element_counts(self, matpower_case14_path):
        """Test that load_net with .m file has correct element counts."""
        from gridfm_datakit.powsybl import load_net

        loaded = load_net(matpower_case14_path)

        assert loaded.gfm_net.buses.shape[0] == 14
        assert len(loaded.pp_net.get_buses()) == 14
        assert loaded.gfm_net.gens.shape[0] == 5


class TestLoadMetadata:
    """Test cases for the load_metadata function."""

    def test_returns_network_metadata(self, matpower_case14_path):
        """Test that load_metadata returns a NetworkMetadata with gen_costs."""
        from gridfm_datakit.powsybl import load_metadata, NetworkMetadata

        metadata = load_metadata(matpower_case14_path)

        assert isinstance(metadata, NetworkMetadata)
        assert isinstance(metadata.gen_costs, dict)
        assert len(metadata.gen_costs) > 0
        assert isinstance(metadata.extra, dict)

    def test_gen_cost_format(self, matpower_case14_path):
        """Test that cost coefficients have correct format."""
        from gridfm_datakit.powsybl import load_metadata

        metadata = load_metadata(matpower_case14_path)

        for gen_idx, costs in metadata.gen_costs.items():
            assert len(costs) >= 1
            for coeff in costs:
                assert isinstance(coeff, (int, float))

    def test_non_matpower_file(self, tmp_path):
        """Test that load_metadata handles non-MATPOWER files gracefully."""
        from gridfm_datakit.powsybl import load_metadata

        dummy_file = tmp_path / "test.json"
        dummy_file.write_text("{}")

        metadata = load_metadata(str(dummy_file))
        assert len(metadata.gen_costs) == 0


class TestToMatFile:
    """Test cases for to_mat_file function."""

    def test_creates_loadable_mat_file(self, tmp_path):
        """Test that to_mat_file creates a .mat file loadable by pypowsybl."""
        from gridfm_datakit.powsybl import to_mat_file
        from gridfm_datakit.network import load_net_from_pglib
        from gridfm_datakit.powsybl import pypowsybl as pp

        net = load_net_from_pglib("case14_ieee")
        mat_path = tmp_path / "test.mat"
        result = to_mat_file(net, mat_path)

        assert result.exists()
        assert result.suffix == ".mat"
        assert isinstance(result, Path)

        pb_net = pp.network.load(str(mat_path))
        assert pb_net is not None

    def test_preserves_element_counts(self, tmp_path):
        """Test that element counts are preserved in conversion."""
        from gridfm_datakit.powsybl import to_mat_file
        from gridfm_datakit.network import load_net_from_pglib
        from gridfm_datakit.powsybl import pypowsybl as pp

        net = load_net_from_pglib("case14_ieee")
        mat_path = tmp_path / "test.mat"
        to_mat_file(net, mat_path)

        pp_net = pp.network.load(str(mat_path))
        assert len(pp_net.get_buses()) == net.buses.shape[0]
        assert len(pp_net.get_generators()) == net.gens.shape[0]


class TestConvertMToMat:
    """Test cases for convert_m_to_mat function."""

    def test_creates_loadable_mat_file(self, matpower_case14_path, tmp_path):
        """Test that convert_m_to_mat creates a .mat file loadable by pypowsybl."""
        from gridfm_datakit.powsybl import convert_m_to_mat
        from gridfm_datakit.powsybl import pypowsybl as pp

        mat_path = tmp_path / "converted.mat"
        result = convert_m_to_mat(matpower_case14_path, mat_path)

        assert result.exists()

        pp_net = pp.network.load(str(mat_path))
        assert pp_net is not None
        assert len(pp_net.get_buses()) == 14

    def test_file_not_found(self, tmp_path):
        """Test that convert_m_to_mat raises FileNotFoundError for missing file."""
        from gridfm_datakit.powsybl import convert_m_to_mat

        with pytest.raises(FileNotFoundError):
            convert_m_to_mat(tmp_path / "nonexistent.m")
