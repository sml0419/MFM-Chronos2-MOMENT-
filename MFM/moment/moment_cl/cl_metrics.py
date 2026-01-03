"""
Continual Learning Evaluation Metrics for Time Series Models.

Implements standard continual learning metrics:
- Backward Transfer (BWT): Measures forgetting on previous domains
- Forward Transfer (FWT): Measures positive transfer to future domains
- Forgetting: Average performance drop on previous domains
- Average Accuracy: Overall performance across all domains
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class PerformanceMatrix:
    """Stores and manages performance across domains in continual learning.

    The performance matrix R has dimensions [n_domains x n_domains], where:
    - R[i, j] is the performance on domain j after training on domains 0...i
    - Diagonal elements R[i, i] are performance immediately after training on domain i
    - Lower triangle R[i, j] (i > j) measures backward transfer (forgetting)
    - Upper triangle R[i, j] (i < j) would measure forward transfer (if evaluated)
    """

    def __init__(self):
        self.matrix = {}  # Dict mapping (train_idx, eval_idx) -> performance

    def add_performance(self, train_idx: int, eval_idx: int, performance: float):
        """Add a performance value to the matrix.

        Args:
            train_idx: Index of domain we've trained up to (0-indexed)
            eval_idx: Index of domain we're evaluating on (0-indexed)
            performance: Performance metric (lower is better, e.g., MSE)
        """
        self.matrix[(train_idx, eval_idx)] = performance

    def get_performance(self, train_idx: int, eval_idx: int) -> float:
        """Get performance for a specific (train_idx, eval_idx) pair."""
        return self.matrix.get((train_idx, eval_idx), None)

    def to_numpy(self, n_domains: int) -> np.ndarray:
        """Convert to numpy array.

        Args:
            n_domains: Total number of domains

        Returns:
            2D numpy array [n_domains x n_domains] with NaN for unavailable values
        """
        arr = np.full((n_domains, n_domains), np.nan)
        for (train_idx, eval_idx), perf in self.matrix.items():
            arr[train_idx, eval_idx] = perf
        return arr

    def to_dataframe(self, n_domains: int, domain_names: List[str] = None) -> pd.DataFrame:
        """Convert to pandas DataFrame for visualization.

        Args:
            n_domains: Total number of domains
            domain_names: Optional list of domain names for labels

        Returns:
            DataFrame with domains as rows/columns
        """
        arr = self.to_numpy(n_domains)

        if domain_names is None:
            domain_names = [f"Domain_{i}" for i in range(n_domains)]

        df = pd.DataFrame(
            arr,
            index=[f"After_{name}" for name in domain_names],
            columns=domain_names
        )
        return df

    def __repr__(self):
        return f"PerformanceMatrix with {len(self.matrix)} entries"


def compute_backward_transfer(perf_matrix: PerformanceMatrix, n_domains: int) -> float:
    """Compute Backward Transfer (BWT) metric.

    BWT measures the influence of learning new domains on performance of previous domains.
    Negative BWT indicates forgetting (catastrophic forgetting).
    Positive BWT indicates positive backward transfer.

    Formula:
        BWT = (1 / (n_domains - 1)) * sum_{i=1}^{n_domains-1} (R[n_domains-1, i] - R[i, i])

    Where:
        R[i, j] is performance on domain j after training on domains 0...i
        R[i, i] is performance on domain i immediately after training on it
        R[n_domains-1, i] is final performance on domain i after training on all domains

    Args:
        perf_matrix: PerformanceMatrix containing all evaluations
        n_domains: Total number of domains

    Returns:
        BWT score (negative = forgetting, positive = positive transfer)
    """
    if n_domains < 2:
        return 0.0

    bwt_sum = 0.0
    count = 0

    for i in range(n_domains - 1):
        # Performance on domain i immediately after training
        initial_perf = perf_matrix.get_performance(i, i)
        # Performance on domain i after training on all domains
        final_perf = perf_matrix.get_performance(n_domains - 1, i)

        if initial_perf is not None and final_perf is not None:
            # For loss metrics (lower is better), forgetting shows as positive difference
            # So we negate to make negative BWT indicate forgetting
            bwt_sum += (initial_perf - final_perf)
            count += 1

    if count == 0:
        return 0.0

    bwt = bwt_sum / count
    return bwt


def compute_forward_transfer(perf_matrix: PerformanceMatrix, n_domains: int,
                             random_baselines: List[float] = None) -> float:
    """Compute Forward Transfer (FWT) metric.

    FWT measures the influence of learning previous domains on learning new domains.
    Positive FWT indicates that previous knowledge helps learn new domains faster.

    Formula:
        FWT = (1 / (n_domains - 1)) * sum_{i=1}^{n_domains-1} (R[i-1, i] - R_random[i])

    Where:
        R[i-1, i] is performance on domain i before training on it (after training on 0...i-1)
        R_random[i] is random baseline performance on domain i

    Note: This requires evaluating on domain i BEFORE training on it, which is often
    skipped in practice. If not available, returns None.

    Args:
        perf_matrix: PerformanceMatrix containing all evaluations
        n_domains: Total number of domains
        random_baselines: List of random/untrained performance on each domain

    Returns:
        FWT score or None if data not available
    """
    if n_domains < 2 or random_baselines is None:
        return None

    fwt_sum = 0.0
    count = 0

    for i in range(1, n_domains):
        # Performance on domain i before training on it
        before_perf = perf_matrix.get_performance(i - 1, i)
        # Random baseline on domain i
        random_perf = random_baselines[i]

        if before_perf is not None and random_perf is not None:
            # Improvement over random baseline (negative = positive FWT for loss metrics)
            fwt_sum += (random_perf - before_perf)
            count += 1

    if count == 0:
        return None

    fwt = fwt_sum / count
    return fwt


def compute_forgetting(perf_matrix: PerformanceMatrix, n_domains: int) -> float:
    """Compute average forgetting across all domains.

    Forgetting measures the maximum performance degradation on each domain.

    Formula:
        Forgetting = (1 / (n_domains - 1)) * sum_{i=0}^{n_domains-2} max_{j in [i, n-1]} (R[j, i] - R[n-1, i])

    This finds the best performance achieved on each domain, then measures drop from that.

    Args:
        perf_matrix: PerformanceMatrix containing all evaluations
        n_domains: Total number of domains

    Returns:
        Average forgetting (positive = forgetting occurred)
    """
    if n_domains < 2:
        return 0.0

    forgetting_sum = 0.0
    count = 0

    for i in range(n_domains - 1):
        # Find best performance on domain i across all training stages
        best_perf = float('inf')
        for j in range(i, n_domains):
            perf = perf_matrix.get_performance(j, i)
            if perf is not None:
                best_perf = min(best_perf, perf)

        # Final performance on domain i
        final_perf = perf_matrix.get_performance(n_domains - 1, i)

        if best_perf != float('inf') and final_perf is not None:
            # For loss metrics, forgetting is positive difference
            forgetting = final_perf - best_perf
            forgetting_sum += max(0, forgetting)  # Only count positive forgetting
            count += 1

    if count == 0:
        return 0.0

    avg_forgetting = forgetting_sum / count
    return avg_forgetting


def compute_average_performance(perf_matrix: PerformanceMatrix, n_domains: int) -> float:
    """Compute average performance across all domains after training on all.

    This is the overall continual learning performance metric.

    Formula:
        Avg_Perf = (1 / n_domains) * sum_{i=0}^{n_domains-1} R[n_domains-1, i]

    Args:
        perf_matrix: PerformanceMatrix containing all evaluations
        n_domains: Total number of domains

    Returns:
        Average performance (lower is better for loss metrics)
    """
    if n_domains == 0:
        return float('inf')

    perf_sum = 0.0
    count = 0

    for i in range(n_domains):
        perf = perf_matrix.get_performance(n_domains - 1, i)
        if perf is not None:
            perf_sum += perf
            count += 1

    if count == 0:
        return float('inf')

    avg_perf = perf_sum / count
    return avg_perf


def compute_all_metrics(perf_matrix: PerformanceMatrix, n_domains: int,
                       random_baselines: List[float] = None) -> Dict[str, float]:
    """Compute all continual learning metrics.

    Args:
        perf_matrix: PerformanceMatrix containing all evaluations
        n_domains: Total number of domains
        random_baselines: Optional list of random baseline performance

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'backward_transfer': compute_backward_transfer(perf_matrix, n_domains),
        'forward_transfer': compute_forward_transfer(perf_matrix, n_domains, random_baselines),
        'forgetting': compute_forgetting(perf_matrix, n_domains),
        'average_performance': compute_average_performance(perf_matrix, n_domains),
    }

    return metrics


def print_metrics_summary(metrics: Dict[str, float], metric_name: str = "MSE"):
    """Print a formatted summary of continual learning metrics.

    Args:
        metrics: Dictionary of computed metrics
        metric_name: Name of the performance metric (e.g., 'MSE', 'MAE')
    """
    print("\n" + "="*60)
    print(f"Continual Learning Metrics Summary ({metric_name})")
    print("="*60)

    print(f"\nAverage Performance: {metrics['average_performance']:.6f}")
    print(f"  → Overall {metric_name} across all domains after sequential training")

    print(f"\nBackward Transfer (BWT): {metrics['backward_transfer']:.6f}")
    if metrics['backward_transfer'] < 0:
        print(f"  → NEGATIVE: Catastrophic forgetting detected!")
        print(f"  → Performance degraded by {abs(metrics['backward_transfer']):.6f} on average")
    elif metrics['backward_transfer'] > 0:
        print(f"  → POSITIVE: Backward knowledge transfer!")
        print(f"  → Performance improved by {metrics['backward_transfer']:.6f} on average")
    else:
        print(f"  → ZERO: No forgetting or backward transfer")

    print(f"\nForgetting: {metrics['forgetting']:.6f}")
    if metrics['forgetting'] > 0:
        print(f"  → Domains forgot {metrics['forgetting']:.6f} {metric_name} on average")
    else:
        print(f"  → No forgetting detected")

    if metrics['forward_transfer'] is not None:
        print(f"\nForward Transfer (FWT): {metrics['forward_transfer']:.6f}")
        if metrics['forward_transfer'] > 0:
            print(f"  → POSITIVE: Previous domains help learning new domains!")
        else:
            print(f"  → NEGATIVE: Previous domains interfere with new learning")
    else:
        print(f"\nForward Transfer (FWT): Not available")
        print(f"  → Requires evaluation before training on each domain")

    print("="*60 + "\n")


def save_metrics_to_csv(perf_matrix: PerformanceMatrix, metrics: Dict[str, float],
                        n_domains: int, domain_names: List[str],
                        save_path: str):
    """Save performance matrix and metrics to CSV files.

    Args:
        perf_matrix: PerformanceMatrix containing all evaluations
        metrics: Dictionary of computed metrics
        n_domains: Total number of domains
        domain_names: List of domain names
        save_path: Base path for saving CSV files
    """
    from pathlib import Path

    # Ensure parent directory exists
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save performance matrix
    df_matrix = perf_matrix.to_dataframe(n_domains, domain_names)
    matrix_path = save_path.replace('.csv', '_matrix.csv')
    df_matrix.to_csv(matrix_path)
    print(f"Performance matrix saved to {matrix_path}")

    # Save metrics summary
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(save_path, index=False)
    print(f"Metrics summary saved to {save_path}")
