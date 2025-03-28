"""
Test runner for all ambulance location models
"""

import os
import sys
import importlib
import time

def run_all_tests():
    """Run all test files in the test directory"""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path so we can import modules
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # List of test modules to run
    test_modules = [
        'test_artm_model',
        'test_dsm_model',
        'test_ertm_model',
        'test_ertm_demand_model',
        'test_osm_graph',
        'test_osm_distance'
    ]
    
    # Optional: Princeton test (takes longer)
    # test_modules.append('test_ertm_princeton')
    
    # Run each test module
    results = {}
    for module_name in test_modules:
        print(f"\n{'='*50}")
        print(f"Running tests in {module_name}...")
        print(f"{'='*50}")
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find the main test function (should be named test_*)
            test_functions = [name for name in dir(module) if name.startswith('test_') and callable(getattr(module, name))]
            
            if test_functions:
                start_time = time.time()
                
                # Run each test function
                for func_name in test_functions:
                    print(f"\nRunning {func_name}...")
                    test_func = getattr(module, func_name)
                    test_func()
                
                end_time = time.time()
                duration = end_time - start_time
                results[module_name] = {'status': 'PASSED', 'duration': duration}
                print(f"\n{module_name} tests completed in {duration:.2f} seconds")
            else:
                print(f"No test functions found in {module_name}")
                results[module_name] = {'status': 'NO TESTS', 'duration': 0}
        
        except Exception as e:
            print(f"Error running tests in {module_name}: {str(e)}")
            results[module_name] = {'status': 'FAILED', 'error': str(e), 'duration': 0}
    
    # Print summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for module_name, result in results.items():
        status = result['status']
        duration = result.get('duration', 0)
        
        if status == 'PASSED':
            print(f"{module_name:30} - {status} ({duration:.2f}s)")
        else:
            all_passed = False
            error = result.get('error', 'Unknown error')
            print(f"{module_name:30} - {status} - {error}")
    
    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. See details above.")

if __name__ == "__main__":
    run_all_tests() 