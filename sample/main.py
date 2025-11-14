"""
Main Execution File
Entry point for running optimization workflows from YAML configs.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from app.optimization_processor import optimization_runner


def main():
    """Main execution function."""
    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'tests/test_data/sample_data/portfolio_optimization/config_tax_loss_harvesting.yaml'

    # Run optimization from YAML config
    result = optimization_runner(config_path)

    # Print result as JSON
    if result.get('status') == 'success' and 'output' in result:
        output = result['output']
        optimization_result = {
            'status': output.get('status', result['status']),
            'sell_decisions': output.get('sell_decisions', []),
            'summary': output.get('optimization_summary', {})
        }
        print("\n" + json.dumps(optimization_result, indent=2, default=str))
    else:
        print("\n" + json.dumps(result, indent=2, default=str))

    return result


if __name__ == "__main__":
    main()