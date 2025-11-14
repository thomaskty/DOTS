from app.models.stats import *
from app.playbooks import *


MODEL_OBJ = {
    'general':MTGeneralRegression,
    'tobit':MTTobitRegression,
    'poisson':MTPoissonRegression,
}

PLAYBOOK_TYPES = {
    'base': BasePlaybook,
    'portfolio_optimization': PortfolioOptimizationPlaybook,
}
