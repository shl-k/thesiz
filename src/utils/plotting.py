import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union


def plot_training_rewards(
    reward_csv_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    window_size: int = 100,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot training rewards from a CSV file with episode and reward columns.
    
    Args:
        reward_csv_path: Path to the CSV file containing episode and reward data
        save_path: Path to save the plot to (optional)
        window_size: Size of the rolling window for smoothing (default: 100)
        figsize: Figure size as (width, height) in inches (default: (12, 6))
        title: Plot title (default: derived from filename)
        show_plot: Whether to display the plot (default: True)
    """
    # Load the reward data
    df = pd.read_csv(reward_csv_path)
    
    # Check that required columns exist
    if 'episode' not in df.columns or 'reward' not in df.columns:
        raise ValueError(f"CSV file must contain 'episode' and 'reward' columns. Found: {df.columns}")
    
    # Create figure and axis
    plt.figure(figsize=figsize)
    
    # Plot raw rewards (semi-transparent)
    plt.plot(df['episode'], df['reward'], color='lightblue', alpha=0.3, label='Raw rewards')
    
    # Calculate and plot rolling average
    rolling_mean = df['reward'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(df['episode'], rolling_mean, color='blue', linewidth=2, 
             label=f'Rolling mean (window={window_size})')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Set title
    if title is None:
        title = f"Training Rewards - {Path(reward_csv_path).stem}"
    plt.title(title)
    
    # Set labels and grid
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Add some statistics
    mean_reward = df['reward'].mean()
    median_reward = df['reward'].median()
    max_reward = df['reward'].max()
    
    recent_window = min(1000, len(df))
    recent_mean = df['reward'].tail(recent_window).mean()
    
    stats_text = (
        f"Overall stats:\n"
        f"Mean: {mean_reward:.1f}\n"
        f"Median: {median_reward:.1f}\n"
        f"Max: {max_reward:.1f}\n"
        f"Recent {recent_window} episodes mean: {recent_mean:.1f}"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_training_runs(
    reward_csv_paths: List[Union[str, Path]],
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    window_size: int = 100,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Comparison of Training Runs",
    show_plot: bool = True
) -> None:
    """
    Compare multiple training runs by plotting their reward curves.
    
    Args:
        reward_csv_paths: List of paths to CSV files containing episode and reward data
        labels: List of labels for each run (default: derived from filenames)
        save_path: Path to save the plot to (optional)
        window_size: Size of the rolling window for smoothing (default: 100)
        figsize: Figure size as (width, height) in inches (default: (12, 6))
        title: Plot title (default: "Comparison of Training Runs")
        show_plot: Whether to display the plot (default: True)
    """
    # Create figure and axis
    plt.figure(figsize=figsize)
    
    # Generate default labels if not provided
    if labels is None:
        labels = [Path(path).stem for path in reward_csv_paths]
    
    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(reward_csv_paths)))
    
    # Plot each run
    for i, (path, label) in enumerate(zip(reward_csv_paths, labels)):
        df = pd.read_csv(path)
        
        # Check that required columns exist
        if 'episode' not in df.columns or 'reward' not in df.columns:
            raise ValueError(f"CSV file must contain 'episode' and 'reward' columns. Found: {df.columns}")
        
        # Calculate rolling average
        rolling_mean = df['reward'].rolling(window=window_size, min_periods=1).mean()
        
        # Plot rolling average
        plt.plot(df['episode'], rolling_mean, color=colors[i], linewidth=2, label=label)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward (Rolling Mean)')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example usage
    # plot_training_rewards("logs/simple_dispatch/reward_log.csv", 
    #                      save_path="logs/simple_dispatch/reward_plot.png")
    pass 