"""
Optimization Processor
Main entry point for running optimization playbooks from configuration files.
"""

import yaml
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any

from app.playbooks.optimization.base_playbook import PlaybookConfig, PlaybookResult
from app.utils.types import get_playbook_class
from app.utils.logger import Logger

# Initialize logger
log_name = os.environ.get('log_name', 'MODEL-OPTIMIZATION-RUN')
log_level = os.environ.get('log_level', 'INFO')
log = Logger(log_name, log_level)
LOGGER = log.logger

# Base output directory
BASE_OUTPUT_DIR = Path(__file__).parent.parent.parent / 'outputs'


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    LOGGER.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    LOGGER.debug(f"Configuration loaded successfully")
    return config


def load_data_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load input data based on configuration"""
    input_data = {}

    # New format: datasets section
    if 'datasets' in config:
        datasets = config['datasets']

        if 'holdings' in datasets:
            LOGGER.info(f"Loading holdings from: {datasets['holdings']}")
            input_data['holdings'] = pd.read_csv(datasets['holdings'])

        if 'acquisitions' in datasets:
            LOGGER.info(f"Loading acquisitions from: {datasets['acquisitions']}")
            input_data['purchase_history'] = pd.read_csv(datasets['acquisitions'])

    # Legacy format: input_data_path + individual files
    elif 'input_data_path' in config:
        data_path = Path(config.get('input_data_path', ''))

        if 'holdings_file' in config:
            holdings_path = data_path / config['holdings_file']
            LOGGER.info(f"Loading holdings from: {holdings_path}")
            input_data['holdings'] = pd.read_csv(holdings_path)

        if 'purchase_history_file' in config:
            purchase_path = data_path / config['purchase_history_file']
            LOGGER.info(f"Loading purchase history from: {purchase_path}")
            input_data['purchase_history'] = pd.read_csv(purchase_path)

    return input_data


def get_output_directory(config: Dict[str, Any]) -> str:
    """Construct output directory path."""
    model_name = config.get('model_name') or config.get('name', 'optimization')

    # Check for base_path in config
    base_path = config.get('base_path')

    if base_path:
        output_dir = Path(base_path) / 'outputs' / model_name
        LOGGER.debug(f"Using base_path from config: {base_path}")
    else:
        if 'output' in config and isinstance(config['output'], dict):
            if 'directory' in config['output'] and Path(config['output']['directory']).is_absolute():
                return config['output']['directory']
        elif 'output_dir' in config and Path(config['output_dir']).is_absolute():
            return config['output_dir']

        output_dir = BASE_OUTPUT_DIR / model_name

    return str(output_dir)

def optimization_runner(config_path: str) -> PlaybookResult:
    """Main runner for optimization playbooks."""
    LOGGER.info("OPTIMIZATION PROCESSOR")

    # Load configuration
    config_dict = load_config(config_path)

    # Get proper output directory
    output_dir = get_output_directory(config_dict)

    # Create playbook config
    playbook_config = PlaybookConfig(
        model_name=config_dict.get('model_name', config_dict.get('name', 'optimization')),
        model_id=config_dict.get('model_id'),
        submission_id=config_dict.get('submission_id'),
        model_type=config_dict.get('model_type', config_dict.get('optimizer_type', 'milp')),
        playbook_type=config_dict.get('playbook_type', 'portfolio_optimization'),
        datasets=config_dict.get('datasets', {}),
        columns=config_dict.get('columns', {}),
        constraints=config_dict.get('constraints', []),
        objective=config_dict.get('objective', {}),
        optimizer_params=config_dict.get('optimizer_params', {}),
        tax_parameters=config_dict.get('tax_parameters', config_dict.get('metadata', {})),
        output={'save_results': config_dict.get('save_results', True), 'directory': output_dir},
        metadata=config_dict.get('metadata', {}),
        name=config_dict.get('name'),
        description=config_dict.get('description', ''),
        input_data_path=config_dict.get('input_data_path'),
        output_dir=output_dir
    )

    LOGGER.info(f"Model: {playbook_config.model_name}")
    if playbook_config.model_id:
        LOGGER.info(f"Model ID: {playbook_config.model_id}")
    if playbook_config.submission_id:
        LOGGER.info(f"Submission ID: {playbook_config.submission_id}")
    LOGGER.info(f"Output directory: {output_dir}")

    # Load input data
    LOGGER.info("Loading input data...")
    input_data = load_data_from_config(config_dict)
    LOGGER.info(f"Loaded {len(input_data)} data sources")
    LOGGER.info(f"Holdings: {len(input_data['holdings'])} tickers")
    LOGGER.info(f"Purchase History: {len(input_data['purchase_history'])} lots")

    # Determine playbook type
    playbook_type = config_dict.get('playbook_type', 'portfolio_optimization')

    LOGGER.info(f"Initializing playbook: {playbook_type}")

    # Create and execute playbook
    try:
        PlaybookClass = get_playbook_class(playbook_type)
        playbook = PlaybookClass(config=playbook_config)
    except ValueError as e:
        LOGGER.error(f"Failed to initialize playbook: {e}")
        raise

    LOGGER.info("EXECUTING PLAYBOOK")

    # Execute
    result = playbook.execute(input_data)

    # Log summary
    LOGGER.info("EXECUTION COMPLETE")
    LOGGER.info(f"Status: {result.status.upper()}")
    LOGGER.info(f"Execution Time: {result.execution_time:.2f}s")

    if result.optimization_result:
        LOGGER.info(f"Optimization Results:")
        LOGGER.info(f"Status: {result.optimization_result.status.value}")
        LOGGER.info(f"Objective Value (Tax Liability): ${result.optimization_result.objective_value:,.2f}")
        LOGGER.info(f"Solver Time: {result.optimization_result.solver_time:.4f}s")

    if result.output_summary and 'optimization_summary' in result.output_summary:
        summary = result.output_summary['optimization_summary']

        LOGGER.info("TAX OPTIMIZATION SUMMARY")

        # Concise one-line format
        LOGGER.info(
            f"ST Gains: ${summary['total_short_term_gain']:,.2f} | ST Losses: ${summary['total_short_term_loss']:,.2f} | ST Net: ${summary['net_short_term_gain']:,.2f}")
        LOGGER.info(
            f"LT Gains: ${summary['total_long_term_gain']:,.2f} | LT Losses: ${summary['total_long_term_loss']:,.2f} | LT Net: ${summary['net_long_term_gain']:,.2f}")
        LOGGER.info(
            f"Total Net Capital Gain: ${summary['total_net_capital_gain']:,.2f} | Taxable: ${summary['taxable_capital_gain']:,.2f}")

        if summary['tax_offset'] > 0:
            LOGGER.info(f"Ordinary Income Offset: ${summary['tax_offset']:,.2f}")
        if summary['carryforward_loss'] > 0:
            LOGGER.info(f"Capital Loss Carryforward: ${summary['carryforward_loss']:,.2f}")

        LOGGER.info(
            f"Tax Rates - ST: {summary['short_term_tax_rate'] * 100:.1f}% | LT: {summary['long_term_tax_rate'] * 100:.1f}%")
        LOGGER.info(f"Estimated Tax: ${summary['estimated_tax']:,.2f}")

    if result.warnings:
        LOGGER.warning(f"Warnings: {len(result.warnings)}")
        for warning in result.warnings:
            LOGGER.warning(f"{warning}")

    if result.errors:
        LOGGER.error(f"Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            LOGGER.error(f"{error}")



    return result