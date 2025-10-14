import sys
import os
from pathlib import Path
import subprocess
import time
import matplotlib.pyplot as plt

# Add the current directory to Python path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

def setup_environment():
    """Set up the experimental environment"""
    
    # Create necessary directories
    directories = ['results', 'figures', 'experiments']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("Environment setup complete.")

def run_validation_suite():
    """Run the complete PTS validation suite"""
    
    print("=" * 60)
    print("RUNNING PHASE-TRANSITIONAL SCALING VALIDATION SUITE")
    print("=" * 60)
    
    try:
        from experiments.pts_validation import run_complete_validation
        
        print("Starting comprehensive validation experiments...")
        start_time = time.time()
        
        results = run_complete_validation()
        
        end_time = time.time()
        print(f"\nValidation completed in {end_time - start_time:.1f} seconds")
        
        return results
        
    except ImportError as e:
        print(f"Error importing validation module: {e}")
        return None
    except Exception as e:
        print(f"Error running validation: {e}")
        return None

def generate_all_figures():
    try:
        from experiments.generate_figures import generate_all_figures as gen_figs
        gen_figs()
        print("All figures generated successfully!")
        
    except ImportError as e:
        print(f"Error importing figure generation module: {e}")
    except Exception as e:
        print(f"Error generating figures: {e}")

def main():    
    # Setup
    setup_environment()
    
    # Run validation experiments
    results = run_validation_suite()
    
    if results is None:
        print("Validation failed. Exiting.")
        return
    
    # Generate figures
    generate_all_figures()
    
if __name__ == "__main__":
    main()
