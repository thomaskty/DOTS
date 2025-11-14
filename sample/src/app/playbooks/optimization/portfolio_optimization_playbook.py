"""
Portfolio Optimization Playbook
Tax-efficient stock liquidation optimizer with short-term/long-term capital gains.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from app.playbooks.optimization.base_playbook import BasePlaybook, PlaybookConfig
from app.models.optimizers.base_optimizer import OptimizationResult, VariableType
from app.models.optimizers.milp import MILPOptimizer


class PortfolioOptimizationPlaybook(BasePlaybook):
    """
    Tax-efficient stock liquidation playbook.

    Optimizes stock liquidation considering:
    - Short-term vs long-term capital gains (different tax rates)
    - Capital loss harvesting and offsetting
    - Ordinary income offset limits
    """

    def __init__(self,
                 config: Union[PlaybookConfig, Dict[str, Any], str],
                 optimizer: Optional[MILPOptimizer] = None):
        super().__init__(config, optimizer)

        # Get tax parameters from new structure or legacy metadata
        tax_params = self.config.tax_parameters or self.config.metadata

        # Tax parameters
        self.short_term_cg_rate = tax_params.get('short_term_cg_rate', 0.37)
        self.long_term_cg_rate = tax_params.get('long_term_cg_rate', 0.20)
        self.ordinary_income_offset_limit = tax_params.get('ordinary_income_offset_limit', 3000)
        self.long_term_threshold_days = tax_params.get('long_term_threshold_days', 365)

        valuation_date_str = tax_params.get('valuation_date', datetime.now())
        if isinstance(valuation_date_str, str):
            self.valuation_date = pd.to_datetime(valuation_date_str)
        else:
            self.valuation_date = pd.to_datetime(valuation_date_str)

        # Column mappings (for flexible column names)
        self.column_map = self.config.columns or {}

        # Data structures
        self.lots: Optional[pd.DataFrame] = None
        self.tickers: Optional[List[str]] = None
        self.lot_variables: Dict[int, Any] = {}

        self.log(f"Initialized with tax rates: ST={self.short_term_cg_rate * 100}%, LT={self.long_term_cg_rate * 100}%")

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate input data"""
        errors = []

        if 'holdings' not in input_data:
            errors.append("Missing 'holdings' DataFrame")
        if 'purchase_history' not in input_data:
            errors.append("Missing 'purchase_history' DataFrame")

        if errors:
            return False, errors

        holdings = input_data['holdings']
        required_cols = ['ticker', 'shares_held', 'current_price', 'total_value']
        missing = [col for col in required_cols if col not in holdings.columns]
        if missing:
            errors.append(f"Holdings missing columns: {missing}")

        purchase_history = input_data['purchase_history']
        required_cols = ['ticker', 'acquisition_date', 'purchase_price', 'shares_purchased']
        missing = [col for col in required_cols if col not in purchase_history.columns]
        if missing:
            errors.append(f"Purchase history missing columns: {missing}")

        return len(errors) == 0, errors

    def preprocess_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data for optimization with tax considerations"""
        holdings = input_data['holdings'].copy()
        purchase_history = input_data['purchase_history'].copy()

        # Convert dates
        purchase_history['acquisition_date'] = pd.to_datetime(purchase_history['acquisition_date'])

        # Sort by acquisition date (FIFO)
        purchase_history = purchase_history.sort_values('acquisition_date').reset_index(drop=True)

        # Add lot index
        purchase_history['lot_id'] = range(len(purchase_history))

        # Get current prices
        current_prices = holdings.set_index('ticker')['current_price'].to_dict()
        purchase_history['current_price'] = purchase_history['ticker'].map(current_prices)

        # Calculate holding period
        purchase_history['holding_period_days'] = (
                self.valuation_date - purchase_history['acquisition_date']
        ).dt.days

        # Classify as long-term (>= threshold days) or short-term
        purchase_history['is_long_term'] = purchase_history['holding_period_days'] >= self.long_term_threshold_days

        # Calculate capital gains/losses per share and total
        purchase_history['capital_gain_per_share'] = (
                purchase_history['current_price'] - purchase_history['purchase_price']
        )
        purchase_history['total_capital_gain'] = (
                purchase_history['capital_gain_per_share'] * purchase_history['shares_purchased']
        )

        # Separate into gains and losses
        purchase_history['is_gain'] = purchase_history['total_capital_gain'] > 0
        purchase_history['is_loss'] = purchase_history['total_capital_gain'] < 0

        # Classify by term and gain/loss
        purchase_history['short_term_gain'] = (
                                                      (~purchase_history['is_long_term']) & purchase_history['is_gain']
                                              ).astype(float) * purchase_history['total_capital_gain']

        purchase_history['short_term_loss'] = (
                                                      (~purchase_history['is_long_term']) & purchase_history['is_loss']
                                              ).astype(float) * purchase_history['total_capital_gain']

        purchase_history['long_term_gain'] = (
                                                     purchase_history['is_long_term'] & purchase_history['is_gain']
                                             ).astype(float) * purchase_history['total_capital_gain']

        purchase_history['long_term_loss'] = (
                                                     purchase_history['is_long_term'] & purchase_history['is_loss']
                                             ).astype(float) * purchase_history['total_capital_gain']

        self.lots = purchase_history
        self.tickers = holdings['ticker'].unique().tolist()

        # Log statistics
        self.log(f"Total lots: {len(purchase_history)}")
        self.log(f"Long-term lots: {purchase_history['is_long_term'].sum()}")
        self.log(f"Short-term lots: {(~purchase_history['is_long_term']).sum()}")
        self.log(f"Lots with gains: {purchase_history['is_gain'].sum()}")
        self.log(f"Lots with losses: {purchase_history['is_loss'].sum()}")

        # Required shares per ticker
        shares_required = holdings.set_index('ticker')['shares_held'].to_dict()

        return {
            'lots': purchase_history,
            'holdings': holdings,
            'shares_required': shares_required,
            'tickers': self.tickers
        }

    def build_optimization_model(self, processed_data: Dict[str, Any]) -> None:
        """Build MILP model for tax-efficient liquidation"""
        lots = processed_data['lots']
        shares_required = processed_data['shares_required']
        tickers = processed_data['tickers']

        self.log(f"Building model with {len(lots)} lots and {len(tickers)} tickers")

        # Decision variables: fraction of each lot to sell (0 to 1)
        for idx, row in lots.iterrows():
            var_name = f"lot_{row['lot_id']}"
            self.lot_variables[row['lot_id']] = self.optimizer.add_variable(
                name=var_name,
                var_type=VariableType.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                description=f"{row['ticker']} lot {row['lot_id']}"
            )

        # Calculate total short-term and long-term gains/losses
        st_gains_expr = sum(
            self.lot_variables[row['lot_id']] * row['short_term_gain']
            for _, row in lots.iterrows()
        )

        st_losses_expr = sum(
            self.lot_variables[row['lot_id']] * row['short_term_loss']
            for _, row in lots.iterrows()
        )

        lt_gains_expr = sum(
            self.lot_variables[row['lot_id']] * row['long_term_gain']
            for _, row in lots.iterrows()
        )

        lt_losses_expr = sum(
            self.lot_variables[row['lot_id']] * row['long_term_loss']
            for _, row in lots.iterrows()
        )

        # Net capital gains (with offsetting)
        # Short-term losses offset short-term gains first
        net_st = st_gains_expr + st_losses_expr  # st_losses is negative

        # Long-term losses offset long-term gains first
        net_lt = lt_gains_expr + lt_losses_expr  # lt_losses is negative

        # Calculate taxable amounts
        # If net ST is positive, tax it at ST rate
        # If net LT is positive, tax it at LT rate
        # Excess losses can offset the other type of gain

        # Objective: Minimize total tax liability
        # Tax = (max(0, net_ST) * ST_rate) + (max(0, net_LT) * LT_rate)
        # For linear programming, we approximate with weighted sum
        tax_liability_expr = (
                (st_gains_expr + st_losses_expr) * self.short_term_cg_rate +
                (lt_gains_expr + lt_losses_expr) * self.long_term_cg_rate
        )

        self.optimizer.set_objective(tax_liability_expr, sense='minimize')

        # Constraints: Must sell exactly required shares per ticker
        for ticker in tickers:
            ticker_lots = lots[lots['ticker'] == ticker]
            constraint_expr = sum(
                self.lot_variables[row['lot_id']] * row['shares_purchased']
                for _, row in ticker_lots.iterrows()
            )

            self.optimizer.add_constraint(
                name=f"shares_required_{ticker}",
                expression=constraint_expr,
                constraint_type='eq',
                rhs=shares_required[ticker],
                description=f"Must sell {shares_required[ticker]} shares of {ticker}"
            )

        self.log(f"Model built: {len(self.lot_variables)} variables, {len(tickers)} constraints")
        self.log("Objective: Minimize tax liability considering ST/LT rates and offsetting")

    def extract_solution(self, opt_result: OptimizationResult) -> Dict[str, Any]:
        """Extract solution from optimization result in senior's format"""
        if opt_result.variables is None:
            return {
                'status': 'error',
                'sell_decisions': [],
                'summary': {},
                'error': 'No solution available'
            }

        # Build sell_decisions list
        sell_decisions = []

        for lot_id in self.lot_variables.keys():
            var_name = f"lot_{lot_id}"
            sell_fraction = opt_result.variables.get(var_name, 0.0)
            lot_info = self.lots[self.lots['lot_id'] == lot_id].iloc[0]

            # Add to sell_decisions (include all lots, not just sold ones)
            sell_decisions.append({
                'ticker': lot_info['ticker'],
                'acquisition_date': str(lot_info['acquisition_date'].date()),
                'quantity': int(lot_info['shares_purchased']),
                'quantity_to_sell': float(sell_fraction * lot_info['shares_purchased']),
                'sell': bool(sell_fraction > 0.001),  # Boolean flag
                'sell_fraction': float(sell_fraction),
                'purchase_price': float(lot_info['purchase_price']),
                'current_price': float(lot_info['current_price']),
                'is_long_term': bool(lot_info['is_long_term']),
                'holding_period_days': int(lot_info['holding_period_days']),
                'capital_gain': float(sell_fraction * lot_info['total_capital_gain'])
            })

        # Calculate summary statistics
        sold_lots = [lot for lot in sell_decisions if lot['sell']]

        total_short_term_gain = sum(
            lot['capital_gain'] for lot in sold_lots
            if not lot['is_long_term'] and lot['capital_gain'] > 0
        )

        total_short_term_loss = sum(
            lot['capital_gain'] for lot in sold_lots
            if not lot['is_long_term'] and lot['capital_gain'] < 0
        )

        total_long_term_gain = sum(
            lot['capital_gain'] for lot in sold_lots
            if lot['is_long_term'] and lot['capital_gain'] > 0
        )

        total_long_term_loss = sum(
            lot['capital_gain'] for lot in sold_lots
            if lot['is_long_term'] and lot['capital_gain'] < 0
        )

        # Calculate net amounts
        net_short_term_gain = total_short_term_gain + total_short_term_loss
        net_long_term_gain = total_long_term_gain + total_long_term_loss

        # Total net capital gain/loss
        total_net_capital_gain = net_short_term_gain + net_long_term_gain

        # Calculate tax offset (loss offsetting ordinary income)
        if total_net_capital_gain < 0:
            tax_offset = min(abs(total_net_capital_gain), self.ordinary_income_offset_limit)
            carryforward_loss = abs(total_net_capital_gain) - tax_offset
            taxable_capital_gain = 0
        else:
            tax_offset = 0
            carryforward_loss = 0
            taxable_capital_gain = total_net_capital_gain

        # Calculate estimated tax
        estimated_tax = (
                max(0, net_short_term_gain) * self.short_term_cg_rate +
                max(0, net_long_term_gain) * self.long_term_cg_rate
        )

        # Build result in senior's format
        result = {
            'status': opt_result.status.value,
            'sell_decisions': sell_decisions,
            'summary': {
                'total_short_term_gain': float(total_short_term_gain),
                'total_short_term_loss': float(total_short_term_loss),
                'total_long_term_gain': float(total_long_term_gain),
                'total_long_term_loss': float(total_long_term_loss),
                'net_short_term_gain': float(net_short_term_gain),
                'net_long_term_gain': float(net_long_term_gain),
                'total_net_capital_gain': float(total_net_capital_gain),
                'tax_offset': float(tax_offset),
                'taxable_capital_gain': float(taxable_capital_gain),
                'carryforward_loss': float(carryforward_loss),
                'estimated_tax': float(estimated_tax),
                'short_term_tax_rate': float(self.short_term_cg_rate),
                'long_term_tax_rate': float(self.long_term_cg_rate)
            }
        }

        return result

    def generate_output(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted output"""
        if 'error' in solution:
            return solution

        summary = solution['summary']
        sell_decisions = solution['sell_decisions']

        # Convert sell_decisions to DataFrame
        sell_decisions_df = pd.DataFrame(sell_decisions)

        # Filter only sold lots
        lots_sold_df = sell_decisions_df[sell_decisions_df['sell'] == True].copy()

        # Summary by ticker
        if len(lots_sold_df) > 0:
            ticker_summary = lots_sold_df.groupby('ticker').agg({
                'quantity_to_sell': 'sum',
                'capital_gain': 'sum'
            }).reset_index()
            ticker_summary.columns = ['ticker', 'shares_sold', 'capital_gain']
        else:
            ticker_summary = pd.DataFrame(columns=['ticker', 'shares_sold', 'capital_gain'])

        output = {
            'status': solution['status'],
            'sell_decisions': sell_decisions,
            'optimization_summary': summary,
            'ticker_summary': ticker_summary.to_dict('records')
        }

        # Save outputs if directory specified
        output_config = self.config.output or {}
        output_dir_str = output_config.get('directory') or self.config.output_dir

        if output_dir_str:
            output_path = Path(output_dir_str)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save all sell decisions (including non-sold lots)
            sell_decisions_df.to_csv(output_path / 'sell_decisions.csv', index=False)

            # Save only sold lots
            if len(lots_sold_df) > 0:
                lots_sold_df.to_csv(output_path / 'lots_sold.csv', index=False)

            # Save ticker summary
            ticker_summary.to_csv(output_path / 'ticker_summary.csv', index=False)

            # Save tax summary
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(output_path / 'tax_summary.csv', index=False)

            # Save the complete result in senior's format as JSON
            import json
            from datetime import datetime

            complete_result = {
                'status': solution['status'],
                'sell_decisions': sell_decisions,
                'summary': summary
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f'optimization_result_{timestamp}.json'

            with open(output_path / result_filename, 'w') as f:
                json.dump(complete_result, indent=2, fp=f, default=str)

            self.log(f"Saved result JSON to {output_path / result_filename}")
            self.log(f"Saved CSV files to {output_path}")

        return output