#!/usr/bin/env python3
"""
Script to visualize training rewards from CSV files.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.utils.plotting import plot_training_rewards, compare_training_runs

def main():
    parser = argparse.ArgumentParser(description="Plot training rewards from CSV files")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single plot command
    single_parser = subparsers.add_parser("single", help="Plot rewards from a single CSV file")
    single_parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    single_parser.add_argument("--save", type=str, help="Path to save the plot", default=None)
    single_parser.add_argument("--window", type=int, help="Window size for rolling average", default=100)
    single_parser.add_argument("--title", type=str, help="Plot title", default=None)
    single_parser.add_argument("--no-show", action="store_true", help="Don't show the plot")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare rewards from multiple CSV files")
    compare_parser.add_argument("csv_paths", nargs="+", type=str, help="Paths to the CSV files")
    compare_parser.add_argument("--labels", nargs="+", type=str, help="Labels for each CSV file")
    compare_parser.add_argument("--save", type=str, help="Path to save the plot", default=None)
    compare_parser.add_argument("--window", type=int, help="Window size for rolling average", default=100)
    compare_parser.add_argument("--title", type=str, help="Plot title", default="Comparison of Training Runs")
    compare_parser.add_argument("--no-show", action="store_true", help="Don't show the plot")
    
    args = parser.parse_args()
    
    if args.command == "single":
        plot_training_rewards(
            args.csv_path,
            save_path=args.save,
            window_size=args.window,
            title=args.title,
            show_plot=not args.no_show
        )
    elif args.command == "compare":
        # Check if labels are provided and match the number of CSV files
        if args.labels and len(args.labels) != len(args.csv_paths):
            print(f"Error: Number of labels ({len(args.labels)}) doesn't match number of CSV files ({len(args.csv_paths)})")
            return 1
        
        compare_training_runs(
            args.csv_paths,
            labels=args.labels,
            save_path=args.save,
            window_size=args.window,
            title=args.title,
            show_plot=not args.no_show
        )
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 