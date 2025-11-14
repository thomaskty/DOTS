"""
Types Registry
Central registry for playbook and optimizer types.
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from app.playbooks.optimization.base_playbook import BasePlaybook
    from app.models.optimizers.base_optimizer import BaseOptimizer

# Playbook type registry
PLAYBOOK_TYPES: Dict[str, Type['BasePlaybook']] = {}

# Optimizer type registry
OPTIMIZER_TYPES: Dict[str, Type['BaseOptimizer']] = {}


def get_playbook_class(playbook_type: str) -> Type['BasePlaybook']:
    """Get playbook class by type string."""
    if playbook_type not in PLAYBOOK_TYPES:
        raise ValueError(
            f"Unknown playbook type: {playbook_type}. "
            f"Available types: {list(PLAYBOOK_TYPES.keys())}"
        )
    return PLAYBOOK_TYPES[playbook_type]


def get_optimizer_class(optimizer_type: str) -> Type['BaseOptimizer']:
    """Get optimizer class by type string."""
    if optimizer_type not in OPTIMIZER_TYPES:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Available types: {list(OPTIMIZER_TYPES.keys())}"
        )
    return OPTIMIZER_TYPES[optimizer_type]


# Initialize registry
def _initialize_registry():
    """Initialize the registry with known types."""
    try:
        from app.playbooks.optimization.base_playbook import BasePlaybook
        from app.playbooks.optimization.portfolio_optimization_playbook import PortfolioOptimizationPlaybook
        from app.models.optimizers.base_optimizer import BaseOptimizer
        from app.models.optimizers.milp import MILPOptimizer

        PLAYBOOK_TYPES['base'] = BasePlaybook
        PLAYBOOK_TYPES['portfolio_optimization'] = PortfolioOptimizationPlaybook

        OPTIMIZER_TYPES['base'] = BaseOptimizer
        OPTIMIZER_TYPES['milp'] = MILPOptimizer

    except ImportError:
        pass


_initialize_registry()