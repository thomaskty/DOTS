"""
Base Optimizer Class
Provides abstract interface for all project_latest techniques.
Supports multiple solver backends including GEKKO.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime

from app.utils.logger import Logger

# Initialize logger
log_name = os.environ.get('log_name', 'MODEL-OPTIMIZATION-RUN')
log_level = os.environ.get('log_level', 'INFO')
log = Logger(log_name, log_level)
LOGGER = log.logger


class OptimizationStatus(Enum):
    """Standardized project_latest solution status"""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"
    NOT_SOLVED = "not_solved"
    TIME_LIMIT = "time_limit"


class VariableType(Enum):
    """Variable types for project_latest"""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"


class ConstraintType(Enum):
    """Constraint types"""
    EQUALITY = "eq"
    LESS_EQUAL = "leq"
    GREATER_EQUAL = "geq"


@dataclass
class Variable:
    """Decision variable definition"""
    name: str
    var_type: VariableType
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    initial_value: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.var_type.value,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'initial_value': self.initial_value,
            'description': self.description
        }


@dataclass
class Constraint:
    """Constraint definition"""
    name: str
    constraint_type: ConstraintType
    expression: Any
    rhs: float
    active: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.constraint_type.value,
            'rhs': self.rhs,
            'active': self.active,
            'description': self.description
        }


@dataclass
class OptimizationResult:
    """Standardized project_latest result container"""
    status: OptimizationStatus
    objective_value: Optional[float]
    variables: Optional[Dict[str, float]]
    dual_values: Optional[Dict[str, float]] = None
    solver_time: float = 0.0
    iterations: Optional[int] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'status': self.status.value,
            'objective_value': self.objective_value,
            'variables': self.variables,
            'dual_values': self.dual_values,
            'solver_time': self.solver_time,
            'iterations': self.iterations,
            'message': self.message,
            'metadata': self.metadata
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export result as JSON"""
        result_json = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(result_json)
        return result_json

    def to_dataframe(self) -> pd.DataFrame:
        """Convert variable results to DataFrame"""
        if self.variables is None:
            return pd.DataFrame()

        return pd.DataFrame([
            {'variable': k, 'value': v}
            for k, v in self.variables.items()
        ])


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    Provides standardized interface for project_latest techniques.
    """

    def __init__(self,
                 name: str = "optimizer",
                 solver: Optional[str] = None,
                 verbose: bool = False,
                 log_file: Optional[str] = None):
        """
        Initialize base optimizer.

        Args:
            name: Name identifier for the optimizer
            solver: Solver backend to use
            verbose: Enable verbose output
            log_file: Path to log file for project_latest details
        """
        self.name = name
        self.solver = solver
        self.verbose = verbose
        self.log_file = log_file

        # Model components
        self.variables: Dict[str, Variable] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.objective = None
        self.optimization_sense = 'minimize'  # 'minimize' or 'maximize'

        # Results
        self.result: Optional[OptimizationResult] = None
        self._model = None

        # Metadata
        self.created_at = datetime.now()
        self.solved_at: Optional[datetime] = None

        # Logging
        self.logs: List[str] = []

    @abstractmethod
    def add_variable(self,
                     name: str,
                     var_type: Union[str, VariableType] = VariableType.CONTINUOUS,
                     lb: Optional[float] = None,
                     ub: Optional[float] = None,
                     initial: Optional[float] = None,
                     description: str = "") -> Any:
        """
        Add decision variable to the model.

        Args:
            name: Variable identifier
            var_type: Type of variable
            lb: Lower bound
            ub: Upper bound
            initial: Initial value/guess
            description: Variable description

        Returns:
            Variable object from underlying solver
        """
        pass

    @abstractmethod
    def add_constraint(self,
                       name: str,
                       expression: Any,
                       constraint_type: Union[str, ConstraintType] = ConstraintType.EQUALITY,
                       rhs: float = 0.0,
                       description: str = "") -> None:
        """
        Add constraint to the model.

        Args:
            name: Constraint identifier
            expression: Mathematical expression
            constraint_type: Type of constraint
            rhs: Right-hand side value
            description: Constraint description
        """
        pass

    @abstractmethod
    def set_objective(self,
                      expression: Any,
                      sense: str = 'minimize') -> None:
        """
        Set objective function.

        Args:
            expression: Objective expression to optimize
            sense: 'minimize' or 'maximize'
        """
        pass

    @abstractmethod
    def solve(self, **kwargs) -> OptimizationResult:
        """
        Solve the project_latest problem.

        Returns:
            OptimizationResult containing solution
        """
        pass

    @abstractmethod
    def get_variable_value(self, var_name: str) -> Optional[float]:
        """
        Get optimized value of a variable.

        Args:
            var_name: Name of variable

        Returns:
            Optimized value or None
        """
        pass

    # Common utility methods

    def add_variables_batch(self,
                            names: List[str],
                            var_type: Union[str, VariableType] = VariableType.CONTINUOUS,
                            lb: Optional[float] = None,
                            ub: Optional[float] = None) -> Dict[str, Any]:
        """Add multiple variables at once"""
        return {
            name: self.add_variable(name, var_type, lb, ub)
            for name in names
        }

    def validate_model(self) -> Tuple[bool, List[str]]:
        """
        Validate model before solving.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not self.variables:
            errors.append("No variables defined in model")

        if self.objective is None:
            errors.append("No objective function defined")

        # Check for unbounded variables in certain problem types
        for var_name, var in self.variables.items():
            if var.lower_bound is None and var.upper_bound is None:
                if var.var_type in [VariableType.CONTINUOUS]:
                    errors.append(f"Variable '{var_name}' is unbounded")

        return len(errors) == 0, errors

    def reset(self) -> None:
        """Reset the optimizer to initial state"""
        self.variables.clear()
        self.constraints.clear()
        self.objective = None
        self.result = None
        self._model = None
        self.logs.clear()
        self.solved_at = None

    def get_solution_summary(self) -> Dict[str, Any]:
        """Get human-readable solution summary"""
        if self.result is None:
            return {"error": "No solution available. Run solve() first."}

        return {
            "optimizer": self.name,
            "status": self.result.status.value,
            "objective_value": self.result.objective_value,
            "solver_time": f"{self.result.solver_time:.4f}s",
            "iterations": self.result.iterations,
            "num_variables": len(self.variables),
            "num_constraints": len(self.constraints),
            "solved_at": self.solved_at.isoformat() if self.solved_at else None
        }

    def get_all_variable_values(self) -> Dict[str, float]:
        """Get all variable values as dictionary"""
        if self.result and self.result.variables:
            return self.result.variables
        return {}

    def export_variables_to_csv(self, filepath: str) -> None:
        """Export variable results to CSV"""
        df = self.result.to_dataframe()
        df.to_csv(filepath, index=False)

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

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model statistics"""
        var_types = {}
        for var in self.variables.values():
            var_type = var.var_type.value
            var_types[var_type] = var_types.get(var_type, 0) + 1

        constraint_types = {}
        for const in self.constraints.values():
            const_type = const.constraint_type.value
            constraint_types[const_type] = constraint_types.get(const_type, 0) + 1

        return {
            "num_variables": len(self.variables),
            "variable_types": var_types,
            "num_constraints": len(self.constraints),
            "constraint_types": constraint_types,
            "has_objective": self.objective is not None,
            "optimization_sense": self.optimization_sense
        }

    def print_model_summary(self) -> None:
        """Print comprehensive model summary"""
        LOGGER.info("=" * 70)
        LOGGER.info(f"OPTIMIZATION MODEL SUMMARY: {self.name}")
        LOGGER.info("=" * 70)

        stats = self.get_model_statistics()
        LOGGER.info(f"Variables: {stats['num_variables']}")
        for var_type, count in stats['variable_types'].items():
            LOGGER.info(f"  - {var_type}: {count}")

        LOGGER.info(f"")
        LOGGER.info(f"Constraints: {stats['num_constraints']}")
        for const_type, count in stats['constraint_types'].items():
            LOGGER.info(f"  - {const_type}: {count}")

        LOGGER.info(f"")
        LOGGER.info(f"Objective: {self.optimization_sense}")
        LOGGER.info(f"Solver: {self.solver}")
        LOGGER.info("=" * 70)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"variables={len(self.variables)}, "
                f"constraints={len(self.constraints)}, "
                f"solver={self.solver})")

    def __str__(self) -> str:
        return self.__repr__()