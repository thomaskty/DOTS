"""
app Main Execution File
Entry point for running optimization workflows.
"""

import sys
import json
import os
from pathlib import Path

# Add src to path BEFORE importing app modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from app.optimization_processor import optimization_runner
from app.utils.logger import Logger


def main():
    """Main execution function"""

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'tests/test_data/sample_data/portfolio_optimization/config3.yaml'

    custom_log_name = sys.argv[2] if len(sys.argv) > 2 else None

    # Initialize logger
    os.environ['log_name'] = custom_log_name if custom_log_name else "MODEL-OPTIMIZATION-RUN"
    log_level = os.environ.get('log_level', 'INFO')
    log = Logger(os.environ['log_name'], log_level)
    LOGGER = log.logger

    LOGGER.info("=" * 70)
    LOGGER.info("PORTFOLIO OPTIMIZATION - EXECUTION START")
    LOGGER.info("=" * 70)

    # Run optimization
    result = optimization_runner(config_path)

    # Print final JSON result to console
    if result.output_summary:
        optimization_result = {
            'status': result.output_summary.get('status', result.status),
            'sell_decisions': result.output_summary.get('sell_decisions', []),
            'summary': result.output_summary.get('optimization_summary', {})
        }
        print("\n" + json.dumps(optimization_result, indent=2, default=str))

    # Flush logs
    base_path = Path(config_path).parent.parent.parent.parent
    log.file_handler_flush(str(base_path))

    return result


if __name__ == "__main__":
    main()