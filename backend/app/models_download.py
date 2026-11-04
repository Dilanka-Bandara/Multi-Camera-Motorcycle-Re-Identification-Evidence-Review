"""Phase 0 placeholder; real model provisioning is required in Phase 8."""

import logging


def main() -> None:
    """Explicitly report that inference weights are not yet provisioned."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Phase 0: model provisioning is not implemented yet")


if __name__ == "__main__":
    main()
