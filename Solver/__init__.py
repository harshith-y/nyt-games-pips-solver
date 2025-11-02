"""
Pips Puzzle Solver Package

A CSP-based solver with heuristic-guided backtracking for Pips puzzles.
"""

from .puzzle import PipsPuzzle, Cell, Region, Domino
from .constraints import ConstraintChecker, HeuristicDetector
from .solver import CSPSolver
from .output import SolutionFormatter

__version__ = "1.0.0"
__all__ = [
    'PipsPuzzle',
    'Cell',
    'Region', 
    'Domino',
    'ConstraintChecker',
    'HeuristicDetector',
    'CSPSolver',
    'SolutionFormatter'
]
