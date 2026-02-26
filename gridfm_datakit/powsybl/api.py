try:
    import pypowsybl

    PYPOWSYBL_AVAILABLE = True
except ImportError:
    PYPOWSYBL_AVAILABLE = False
    pypowsybl = None

def check_powsybl_available() -> None:
    """Check if pypowsybl is available, raise ImportError if not."""
    if not PYPOWSYBL_AVAILABLE:
        raise ImportError(
            "pypowsybl is required for this functionality. "
            "Install it with: pip install gridfm-datakit[powsybl]"
        )
