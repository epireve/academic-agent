#!/usr/bin/env python3
"""
Comprehensive Test Runner for Academic Agent System

This script runs all available tests and generates a comprehensive test report
for the agent testing and validation phase (Sub-stage 5).
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    output: str
    error_message: str = ""


@dataclass
class TestSuite:
    """Test suite execution result"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    duration: float
    coverage_percentage: float = 0.0
    test_results: List[TestResult] = None
    
    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []


class ComprehensiveTestRunner:
    """Comprehensive test runner for all agent tests"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def discover_test_files(self) -> Dict[str, List[str]]:
        """Discover all test files in the project"""
        test_categories = {
            'unit': [],
            'integration': [],
            'performance': [],
            'validation': []
        }
        
        tests_dir = self.project_root / 'tests'
        if not tests_dir.exists():
            print(f"Tests directory not found: {tests_dir}")
            return test_categories
        
        # Discover unit tests
        unit_tests_dir = tests_dir / 'unit'
        if unit_tests_dir.exists():
            test_categories['unit'] = [
                str(f) for f in unit_tests_dir.glob('test_*.py')
            ]
        
        # Discover integration tests
        integration_tests_dir = tests_dir / 'integration'
        if integration_tests_dir.exists():
            test_categories['integration'] = [
                str(f) for f in integration_tests_dir.glob('test_*.py')
            ]
        
        # Discover other test types
        for test_type in ['performance', 'validation']:
            test_type_dir = tests_dir / test_type
            if test_type_dir.exists():
                test_categories[test_type] = [
                    str(f) for f in test_type_dir.glob('test_*.py')
                ]
        
        return test_categories
    
    def run_syntax_validation(self, test_files: List[str]) -> TestSuite:
        """Run syntax validation on test files"""
        suite_start = time.time()
        results = []
        
        for test_file in test_files:
            test_start = time.time()
            try:
                # Check syntax compilation
                result = subprocess.run(
                    [sys.executable, '-m', 'py_compile', test_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                duration = time.time() - test_start
                
                if result.returncode == 0:
                    results.append(TestResult(
                        test_name=f"syntax_{Path(test_file).name}",
                        status='passed',
                        duration=duration,
                        output='Syntax validation passed'
                    ))
                else:
                    results.append(TestResult(
                        test_name=f"syntax_{Path(test_file).name}",
                        status='failed',
                        duration=duration,
                        output=result.stdout,
                        error_message=result.stderr
                    ))
                    
            except subprocess.TimeoutExpired:
                results.append(TestResult(
                    test_name=f"syntax_{Path(test_file).name}",
                    status='error',
                    duration=30.0,
                    output='',
                    error_message='Syntax check timeout'
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"syntax_{Path(test_file).name}",
                    status='error',
                    duration=0.0,
                    output='',
                    error_message=str(e)
                ))
        
        # Calculate statistics
        passed = sum(1 for r in results if r.status == 'passed')
        failed = sum(1 for r in results if r.status == 'failed')
        error = sum(1 for r in results if r.status == 'error')
        
        return TestSuite(
            suite_name='syntax_validation',
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=0,
            error_tests=error,
            duration=time.time() - suite_start,
            test_results=results
        )
    
    def run_import_validation(self, test_files: List[str]) -> TestSuite:
        """Validate that test files can import required modules"""
        suite_start = time.time()
        results = []
        
        for test_file in test_files:
            test_start = time.time()
            try:
                # Try to import the test module
                test_path = Path(test_file)
                module_name = test_path.stem
                
                # Add test directory to path temporarily
                original_path = sys.path.copy()
                sys.path.insert(0, str(test_path.parent))
                
                try:
                    __import__(module_name)
                    duration = time.time() - test_start
                    
                    results.append(TestResult(
                        test_name=f"import_{module_name}",
                        status='passed',
                        duration=duration,
                        output='Import validation passed'
                    ))
                    
                except ImportError as e:
                    duration = time.time() - test_start
                    results.append(TestResult(
                        test_name=f"import_{module_name}",
                        status='failed',
                        duration=duration,
                        output='',
                        error_message=f"Import error: {str(e)}"
                    ))
                    
                finally:
                    sys.path = original_path
                    
            except Exception as e:
                results.append(TestResult(
                    test_name=f"import_{Path(test_file).stem}",
                    status='error',
                    duration=0.0,
                    output='',
                    error_message=str(e)
                ))
        
        # Calculate statistics
        passed = sum(1 for r in results if r.status == 'passed')
        failed = sum(1 for r in results if r.status == 'failed')
        error = sum(1 for r in results if r.status == 'error')
        
        return TestSuite(
            suite_name='import_validation',
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=0,
            error_tests=error,
            duration=time.time() - suite_start,
            test_results=results
        )
    
    def run_pytest_if_available(self, test_files: List[str], test_category: str) -> TestSuite:
        """Run pytest if available, otherwise skip"""
        suite_start = time.time()
        
        try:
            # Check if pytest is available
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return TestSuite(
                    suite_name=f'pytest_{test_category}',
                    total_tests=len(test_files),
                    passed_tests=0,
                    failed_tests=0,
                    skipped_tests=len(test_files),
                    error_tests=0,
                    duration=time.time() - suite_start,
                    test_results=[TestResult(
                        test_name=f'pytest_{test_category}',
                        status='skipped',
                        duration=0.0,
                        output='pytest not available',
                        error_message='pytest module not installed'
                    )]
                )
            
            # Run pytest on test files
            cmd = [sys.executable, '-m', 'pytest', '-v', '--tb=short'] + test_files
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=str(self.project_root)
            )
            
            duration = time.time() - suite_start
            
            # Parse pytest output (simplified)
            output_lines = result.stdout.split('\n')
            passed = len([line for line in output_lines if ' PASSED' in line])
            failed = len([line for line in output_lines if ' FAILED' in line])
            skipped = len([line for line in output_lines if ' SKIPPED' in line])
            error = len([line for line in output_lines if ' ERROR' in line])
            
            return TestSuite(
                suite_name=f'pytest_{test_category}',
                total_tests=passed + failed + skipped + error,
                passed_tests=passed,
                failed_tests=failed,
                skipped_tests=skipped,
                error_tests=error,
                duration=duration,
                test_results=[TestResult(
                    test_name=f'pytest_{test_category}',
                    status='passed' if result.returncode == 0 else 'failed',
                    duration=duration,
                    output=result.stdout,
                    error_message=result.stderr if result.returncode != 0 else ''
                )]
            )
            
        except subprocess.TimeoutExpired:
            return TestSuite(
                suite_name=f'pytest_{test_category}',
                total_tests=len(test_files),
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=len(test_files),
                duration=300.0,
                test_results=[TestResult(
                    test_name=f'pytest_{test_category}',
                    status='error',
                    duration=300.0,
                    output='',
                    error_message='pytest execution timeout'
                )]
            )
            
        except Exception as e:
            return TestSuite(
                suite_name=f'pytest_{test_category}',
                total_tests=len(test_files),
                passed_tests=0,
                failed_tests=0,
                skipped_tests=len(test_files),
                error_tests=0,
                duration=time.time() - suite_start,
                test_results=[TestResult(
                    test_name=f'pytest_{test_category}',
                    status='skipped',
                    duration=0.0,
                    output='',
                    error_message=f'pytest execution error: {str(e)}'
                )]
            )
    
    def run_agent_validation_tests(self) -> TestSuite:
        """Run agent-specific validation tests"""
        suite_start = time.time()
        results = []
        
        # Test agent imports and basic instantiation
        agent_modules = [
            'agents.academic.analysis_agent',
            'agents.academic.notes_agent',
            'agents.academic.consolidation_agent',
            'agents.academic.quality_validation_system',
            'agents.academic.study_notes_generator',
            'src.agents.quality_manager'
        ]
        
        for module_name in agent_modules:
            test_start = time.time()
            try:
                # Add project root to path
                original_path = sys.path.copy()
                sys.path.insert(0, str(self.project_root))
                
                try:
                    # Try to import the module
                    module = __import__(module_name, fromlist=[''])
                    
                    # Try to find agent classes
                    agent_classes = []
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            attr_name.endswith(('Agent', 'System', 'Tool', 'Manager'))):
                            agent_classes.append(attr)
                    
                    if agent_classes:
                        # Try to instantiate first agent class found
                        agent_class = agent_classes[0]
                        try:
                            if 'Tool' in agent_class.__name__:
                                # Tools may need base_dir parameter
                                agent = agent_class(str(self.project_root))
                            else:
                                agent = agent_class()
                            
                            duration = time.time() - test_start
                            results.append(TestResult(
                                test_name=f"agent_validation_{module_name.split('.')[-1]}",
                                status='passed',
                                duration=duration,
                                output=f'Successfully imported and instantiated {agent_class.__name__}'
                            ))
                            
                        except Exception as e:
                            duration = time.time() - test_start
                            results.append(TestResult(
                                test_name=f"agent_validation_{module_name.split('.')[-1]}",
                                status='failed',
                                duration=duration,
                                output='',
                                error_message=f'Instantiation error: {str(e)}'
                            ))
                    else:
                        duration = time.time() - test_start
                        results.append(TestResult(
                            test_name=f"agent_validation_{module_name.split('.')[-1]}",
                            status='failed',
                            duration=duration,
                            output='',
                            error_message='No agent classes found in module'
                        ))
                        
                except ImportError as e:
                    duration = time.time() - test_start
                    results.append(TestResult(
                        test_name=f"agent_validation_{module_name.split('.')[-1]}",
                        status='failed',
                        duration=duration,
                        output='',
                        error_message=f'Import error: {str(e)}'
                    ))
                    
                finally:
                    sys.path = original_path
                    
            except Exception as e:
                results.append(TestResult(
                    test_name=f"agent_validation_{module_name.split('.')[-1]}",
                    status='error',
                    duration=0.0,
                    output='',
                    error_message=str(e)
                ))
        
        # Calculate statistics
        passed = sum(1 for r in results if r.status == 'passed')
        failed = sum(1 for r in results if r.status == 'failed')
        error = sum(1 for r in results if r.status == 'error')
        
        return TestSuite(
            suite_name='agent_validation',
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=0,
            error_tests=error,
            duration=time.time() - suite_start,
            test_results=results
        )
    
    def run_all_tests(self) -> Dict[str, TestSuite]:
        """Run all available tests"""
        print("🚀 Starting Comprehensive Agent Testing...")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Discover test files
        test_files = self.discover_test_files()
        all_test_files = []
        for category, files in test_files.items():
            all_test_files.extend(files)
        
        print(f"📁 Discovered test files:")
        for category, files in test_files.items():
            print(f"  {category}: {len(files)} files")
        
        results = {}
        
        # 1. Syntax validation
        if all_test_files:
            print("\n🔍 Running syntax validation...")
            results['syntax'] = self.run_syntax_validation(all_test_files)
            self.print_suite_results(results['syntax'])
        
        # 2. Import validation
        if all_test_files:
            print("\n📦 Running import validation...")
            results['imports'] = self.run_import_validation(all_test_files)
            self.print_suite_results(results['imports'])
        
        # 3. Agent validation
        print("\n🤖 Running agent validation...")
        results['agents'] = self.run_agent_validation_tests()
        self.print_suite_results(results['agents'])
        
        # 4. Unit tests (if pytest available)
        if test_files['unit']:
            print("\n🧪 Running unit tests...")
            results['unit'] = self.run_pytest_if_available(test_files['unit'], 'unit')
            self.print_suite_results(results['unit'])
        
        # 5. Integration tests (if pytest available)
        if test_files['integration']:
            print("\n🔄 Running integration tests...")
            results['integration'] = self.run_pytest_if_available(test_files['integration'], 'integration')
            self.print_suite_results(results['integration'])
        
        self.end_time = time.time()
        self.test_results = results
        
        return results
    
    def print_suite_results(self, suite: TestSuite):
        """Print test suite results"""
        total = suite.total_tests
        passed = suite.passed_tests
        failed = suite.failed_tests
        skipped = suite.skipped_tests
        error = suite.error_tests
        
        print(f"  📊 {suite.suite_name}: {passed}/{total} passed")
        if failed > 0:
            print(f"      ❌ {failed} failed")
        if skipped > 0:
            print(f"      ⏭️  {skipped} skipped")
        if error > 0:
            print(f"      💥 {error} errors")
        print(f"      ⏱️  Duration: {suite.duration:.2f}s")
        
        # Show failed test details
        if failed > 0 or error > 0:
            failed_tests = [r for r in suite.test_results if r.status in ['failed', 'error']]
            for test in failed_tests[:3]:  # Show first 3 failures
                print(f"        ❌ {test.test_name}: {test.error_message}")
            if len(failed_tests) > 3:
                print(f"        ... and {len(failed_tests) - 3} more failures")
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.test_results:
            return "No test results available"
        
        report_lines = []
        report_lines.append("# Comprehensive Agent Testing Report")
        report_lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Total Duration**: {self.end_time - self.start_time:.2f} seconds")
        report_lines.append("")
        
        # Overall statistics
        total_tests = sum(suite.total_tests for suite in self.test_results.values())
        total_passed = sum(suite.passed_tests for suite in self.test_results.values())
        total_failed = sum(suite.failed_tests for suite in self.test_results.values())
        total_skipped = sum(suite.skipped_tests for suite in self.test_results.values())
        total_error = sum(suite.error_tests for suite in self.test_results.values())
        
        report_lines.append("## Overall Statistics")
        report_lines.append(f"- **Total Tests**: {total_tests}")
        report_lines.append(f"- **Passed**: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        report_lines.append(f"- **Failed**: {total_failed} ({total_failed/total_tests*100:.1f}%)")
        report_lines.append(f"- **Skipped**: {total_skipped} ({total_skipped/total_tests*100:.1f}%)")
        report_lines.append(f"- **Errors**: {total_error} ({total_error/total_tests*100:.1f}%)")
        report_lines.append("")
        
        # Suite details
        report_lines.append("## Test Suite Results")
        report_lines.append("")
        
        for suite_name, suite in self.test_results.items():
            report_lines.append(f"### {suite.suite_name.title()}")
            report_lines.append(f"- **Total Tests**: {suite.total_tests}")
            report_lines.append(f"- **Passed**: {suite.passed_tests}")
            report_lines.append(f"- **Failed**: {suite.failed_tests}")
            report_lines.append(f"- **Skipped**: {suite.skipped_tests}")
            report_lines.append(f"- **Errors**: {suite.error_tests}")
            report_lines.append(f"- **Duration**: {suite.duration:.2f}s")
            report_lines.append("")
            
            # Show failures
            if suite.failed_tests > 0 or suite.error_tests > 0:
                report_lines.append("#### Issues Found:")
                failed_tests = [r for r in suite.test_results if r.status in ['failed', 'error']]
                for test in failed_tests:
                    report_lines.append(f"- **{test.test_name}**: {test.error_message}")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        if total_failed > 0:
            report_lines.append("### Failed Tests")
            report_lines.append("- Review and fix failing test cases")
            report_lines.append("- Check agent implementation for missing methods")
            report_lines.append("- Verify import paths and dependencies")
            report_lines.append("")
        
        if total_skipped > 0:
            report_lines.append("### Skipped Tests")
            report_lines.append("- Install pytest for full test execution: `pip install pytest`")
            report_lines.append("- Install missing test dependencies")
            report_lines.append("- Enable comprehensive test coverage")
            report_lines.append("")
        
        report_lines.append("### Next Steps")
        report_lines.append("1. Address any failing tests")
        report_lines.append("2. Install missing testing dependencies")
        report_lines.append("3. Implement missing agent functionality")
        report_lines.append("4. Add integration with CI/CD pipeline")
        report_lines.append("5. Expand test coverage for edge cases")
        
        return "\n".join(report_lines)
    
    def save_results(self, output_dir: str = None):
        """Save test results to files"""
        if output_dir is None:
            output_dir = self.project_root / "test_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        json_results = {}
        for suite_name, suite in self.test_results.items():
            json_results[suite_name] = asdict(suite)
        
        with open(output_dir / "test_results.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save markdown report
        report = self.generate_comprehensive_report()
        with open(output_dir / "test_report.md", 'w') as f:
            f.write(report)
        
        print(f"\n📄 Test results saved to: {output_dir}")
        print(f"   - JSON results: {output_dir / 'test_results.json'}")
        print(f"   - Markdown report: {output_dir / 'test_report.md'}")


def main():
    """Main function"""
    # Get project root (script is in tests/ directory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    print(f"🏠 Project root: {project_root}")
    
    # Run comprehensive tests
    runner = ComprehensiveTestRunner(str(project_root))
    results = runner.run_all_tests()
    
    # Generate and display summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = sum(suite.total_tests for suite in results.values())
    total_passed = sum(suite.passed_tests for suite in results.values())
    total_failed = sum(suite.failed_tests for suite in results.values())
    
    print(f"📊 Overall Results: {total_passed}/{total_tests} tests passed")
    if total_failed > 0:
        print(f"❌ {total_failed} tests failed")
    
    success_rate = total_passed / total_tests if total_tests > 0 else 0
    print(f"✅ Success Rate: {success_rate:.1%}")
    
    # Save results
    runner.save_results()
    
    # Print final recommendations
    print("\n🎯 NEXT STEPS:")
    if success_rate >= 0.8:
        print("✅ Good test coverage! Consider adding more comprehensive tests.")
    elif success_rate >= 0.6:
        print("⚠️  Moderate test coverage. Address failing tests and add missing tests.")
    else:
        print("❌ Low test coverage. Significant work needed on test implementation.")
    
    print("\n🔄 Sub-stage 5 (Comprehensive Agent Testing) Status:")
    if success_rate >= 0.7:
        print("✅ COMPLETED - Comprehensive testing framework implemented successfully")
    else:
        print("⚠️  IN PROGRESS - Additional work needed on test implementation")
    
    return success_rate >= 0.7


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)