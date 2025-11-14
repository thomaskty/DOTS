"""
Base Playbook for Optimization Workflows
Provides abstract interface for creating reusable optimization playbooks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import time
import os
from pathlib import Path
from datetime import datetime

from app.models.optimizers.base_optimizer import BaseOptimizer, OptimizationResult, OptimizationStatus
from app.utils.logger import Logger

# Initialize logger
log_name = os.environ.get('log_name', 'MODEL-OPTIMIZATION-RUN')
log_level = os.environ.get('log_level', 'INFO')
log = Logger(log_name, log_level)
LOGGER = log.logger

@dataclass
class PlaybookConfig:
    """Configuration container for playbook execution"""
    # Model metadata
    model_name: str
    model_id: Optional[str] = None
    submission_id: Optional[str] = None
    model_type: str = "milp"

    # Playbook type
    playbook_type: str = "portfolio_optimization"

    # Datasets
    datasets: Dict[str, str] = field(default_factory=dict)

    # Column mappings
    columns: Dict[str, str] = field(default_factory=dict)

    # Constraints configuration
    constraints: List[Dict[str, Any]] = field(default_factory=list)

    # Objective configuration
    objective: Dict[str, Any] = field(default_factory=dict)

    # Optimizer parameters
    optimizer_params: Dict[str, Any] = field(default_factory=dict)

    # Tax parameters (for portfolio optimization)
    tax_parameters: Dict[str, Any] = field(default_factory=dict)

    # Output configuration
    output: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy support
    name: Optional[str] = None
    description: str = ""
    optimizer_type: Optional[str] = None
    input_data_path: Optional[str] = None
    output_dir: Optional[str] = None
    save_results: bool = True

    def __post_init__(self):
        """Handle legacy config format"""
        # Map legacy fields to new structure
        if self.name and not self.model_name:
            self.model_name = self.name

        if self.optimizer_type and not self.model_type:
            self.model_type = self.optimizer_type

        if not self.output:
            self.output = {
                'save_results': self.save_results,
                'formats': ['json', 'csv']
            }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PlaybookConfig':
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'submission_id': self.submission_id,
            'model_type': self.model_type,
            'playbook_type': self.playbook_type,
            'datasets': self.datasets,
            'columns': self.columns,
            'constraints': self.constraints,
            'objective': self.objective,
            'optimizer_params': self.optimizer_params,
            'tax_parameters': self.tax_parameters,
            'output': self.output,
            'metadata': self.metadata
        }


@dataclass
class PlaybookResult:
    """Standardized playbook execution result"""
    playbook_name: str
    status: str
    optimization_result: Optional[OptimizationResult]
    execution_time: float
    input_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'playbook_name': self.playbook_name,
            'status': self.status,
            'optimization_result': self.optimization_result.to_dict() if self.optimization_result else None,
            'execution_time': self.execution_time,
            'input_summary': self.input_summary,
            'output_summary': self.output_summary,
            'warnings': self.warnings,
            'errors': self.errors,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export result as JSON"""
        result_json = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(result_json)
        return result_json

    def save(self, directory: str, filename: Optional[str] = None) -> str:
        """Save result to directory"""
        Path(directory).mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.playbook_name}_{timestamp}.json"

        filepath = Path(directory) / filename
        self.to_json(str(filepath))
        return str(filepath)


class BasePlaybook(ABC):
    """
    Abstract base class for optimization playbooks.
    """

    def __init__(self,
                 config: Union[PlaybookConfig, Dict[str, Any], str],
                 optimizer: Optional[BaseOptimizer] = None):
        """
        Initialize playbook.

        Args:
            config: PlaybookConfig object, dict, or path to YAML config
            optimizer: Pre-configured optimizer instance (optional)
        """
        if isinstance(config, str):
            self.config = PlaybookConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = PlaybookConfig.from_dict(config)
        else:
            self.config = config

        self.optimizer = optimizer
        self.input_data: Optional[Dict[str, Any]] = None
        self.processed_data: Optional[Dict[str, Any]] = None
        self.result: Optional[PlaybookResult] = None

        self.logs: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate input data"""
        pass

    @abstractmethod
    def preprocess_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess and transform input data"""
        pass

    @abstractmethod
    def build_optimization_model(self, processed_data: Dict[str, Any]) -> None:
        """Build the optimization model"""
        pass

    @abstractmethod
    def extract_solution(self, opt_result: OptimizationResult) -> Dict[str, Any]:
        """Extract and format the optimization solution"""
        pass

    @abstractmethod
    def generate_output(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final output from solution"""
        pass

    def execute(self, input_data: Dict[str, Any]) -> PlaybookResult:
        """Execute the complete playbook workflow"""
        start_time = time.time()
        self.input_data = input_data

        try:
            self.log("Step 1: Validating input data...")
            is_valid, validation_errors = self.validate_input(input_data)

            if not is_valid:
                self.errors.extend(validation_errors)
                return self._create_failed_result(
                    execution_time=time.time() - start_time,
                    error_message=f"Input validation failed: {', '.join(validation_errors)}"
                )

            self.log("✓ Input validation passed")

            self.log("Step 2: Preprocessing data...")
            self.processed_data = self.preprocess_data(input_data)
            self.log("✓ Data preprocessing complete")

            if self.optimizer is None:
                self.log("Step 3: Initializing optimizer...")
                self.optimizer = self._create_optimizer()
                self.log("✓ Optimizer initialized")

            self.log("Step 4: Building optimization model...")
            self.build_optimization_model(self.processed_data)
            self.log("✓ Optimization model built")

            self.log("Step 5: Solving optimization problem...")
            opt_result = self.optimizer.solve()
            self.log(f"✓ Optimization complete - Status: {opt_result.status.value}")

            if opt_result.status not in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
                self.warnings.append(f"Non-optimal solution: {opt_result.status.value}")

            self.log("Step 6: Extracting solution...")
            solution = self.extract_solution(opt_result)
            self.log("✓ Solution extracted")

            self.log("Step 7: Generating output...")
            output = self.generate_output(solution)
            self.log("✓ Output generated")

            execution_time = time.time() - start_time
            self.result = PlaybookResult(
                playbook_name=self.config.name,
                status='success' if opt_result.status == OptimizationStatus.OPTIMAL else 'partial',
                optimization_result=opt_result,
                execution_time=execution_time,
                input_summary=self._summarize_input(input_data),
                output_summary=output,
                warnings=self.warnings,
                errors=self.errors,
                metadata={'config': self.config.to_dict()}
            )

            if self.config.save_results and self.config.output_dir:
                filepath = self.result.save(self.config.output_dir)
                self.log(f"Results saved to: {filepath}")

            return self.result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Playbook execution failed: {str(e)}"
            self.log(error_msg, level="ERROR")
            self.errors.append(error_msg)

            import traceback
            self.errors.append(traceback.format_exc())

            return self._create_failed_result(
                execution_time=execution_time,
                error_message=error_msg
            )

    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer instance based on config"""
        from app.models.optimizers.milp import MILPOptimizer

        optimizer_type = (self.config.model_type or self.config.optimizer_type or 'milp').lower()

        if optimizer_type == 'milp':
            return MILPOptimizer(**self.config.optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create input data summary"""
        summary = {}
        for key, value in input_data.items():
            if isinstance(value, pd.DataFrame):
                summary[key] = {'type': 'DataFrame', 'shape': value.shape}
            elif isinstance(value, (list, np.ndarray)):
                summary[key] = {'type': type(value).__name__, 'length': len(value)}
            else:
                summary[key] = {'type': type(value).__name__}
        return summary

    def _create_failed_result(self, execution_time: float, error_message: str) -> PlaybookResult:
        """Create a failed result object"""
        return PlaybookResult(
            playbook_name=self.config.name,
            status='failed',
            optimization_result=None,
            execution_time=execution_time,
            input_summary=self._summarize_input(self.input_data) if self.input_data else {},
            output_summary={},
            warnings=self.warnings,
            errors=self.errors,
            metadata={'error': error_message}
        )

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)

        if level.upper() == "ERROR":
            LOGGER.error(message)
        elif level.upper() == "WARNING":
            LOGGER.warning(message)
        elif level.upper() == "DEBUG":
            LOGGER.debug(message)
        else:
            LOGGER.info(message)

    def reset(self) -> None:
        """Reset playbook to initial state"""
        self.input_data = None
        self.processed_data = None
        self.result = None
        self.logs.clear()
        self.warnings.clear()
        self.errors.clear()

        if self.optimizer:
            self.optimizer.reset()