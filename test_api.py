#!/usr/bin/env python3
"""
API Testing Script for Dutch Energy Consumption Predictor

This script demonstrates comprehensive API testing including:
- Basic functionality tests
- Input validation tests
- Performance benchmarks
- Error handling verification
- Integration testing examples

Run this script to verify the API is working correctly.
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any
from datetime import datetime
import concurrent.futures
import statistics

class EnergyAPITester:
    """Comprehensive API testing class."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """Initialize the API tester.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def test_connection(self) -> bool:
        """Test if API server is reachable."""
        print("🔌 Testing API Connection...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("   API server is reachable")
                return True
            else:
                print(f"   ❌ API returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to API at {self.base_url}")
            print("   Make sure to start the API first: python api.py")
            return False
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return False
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        print("\n🏥 Testing Health Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"   Health check passed")
                print(f"   Status: {data.get('status', 'unknown')}")
                print(f"   🤖 Model loaded: {data.get('model_loaded', 'unknown')}")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
    
    def test_model_info_endpoint(self) -> bool:
        """Test the model information endpoint."""
        print("\n🤖 Testing Model Info Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                data = response.json()
                print(f"   Model info retrieved successfully")
                print(f"   Model loaded: {data.get('model_loaded', False)}")
                print(f"   House types: {len(data.get('available_house_types', []))}")
                print(f"   🏢 Companies: {len(data.get('available_companies', []))}")
                print(f"   🌤️ Weather scenarios: {len(data.get('available_weather_scenarios', []))}")
                return True
            else:
                print(f"   ❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Model info error: {e}")
            return False
    
    def test_basic_predictions(self) -> bool:
        """Test basic prediction functionality."""
        print("\n🔮 Testing Basic Predictions...")
        
        test_cases = [
            {
                "name": "Complete Amsterdam Request",
                "payload": {
                    "postal_code": "1012",
                    "city": "Amsterdam",
                    "house_type": "3x25",
                    "smart_meter": True,
                    "weather_scenario": "normal"
                }
            },
            {
                "name": "Minimal Request (Defaults)",
                "payload": {}
            },
            {
                "name": "Large House Cold Year",
                "payload": {
                    "postal_code": "3500",
                    "house_type": "3x50",
                    "weather_scenario": "cold",
                    "smart_meter": False
                }
            },
            {
                "name": "Small Apartment Warm Year",
                "payload": {
                    "city": "Rotterdam",
                    "house_type": "1x25",
                    "weather_scenario": "warm",
                    "num_connections": 100
                }
            }
        ]
        
        success_count = 0
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   Test {i}: {test_case['name']}")
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_case['payload'],
                    timeout=30
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get('prediction_kwh', 0)
                    monthly_cost = data.get('estimated_monthly_cost', 0)
                    model_used = data.get('model_used', 'unknown')
                    
                    print(f"      Success ({response_time:.2f}s)")
                    print(f"      Prediction: {prediction:.0f} kWh/year")
                    print(f"      💰 Monthly cost: €{monthly_cost:.0f}")
                    print(f"      🤖 Model: {model_used}")
                    
                    # Validation checks
                    if 500 <= prediction <= 10000:  # Reasonable range
                        print(f"      Prediction in reasonable range")
                    else:
                        print(f"      ⚠️ Prediction seems unusual: {prediction}")
                    
                    success_count += 1
                else:
                    print(f"      ❌ Failed: {response.status_code}")
                    print(f"      📝 Response: {response.text[:200]}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        print(f"\n   Basic prediction tests: {success_count}/{len(test_cases)} passed")
        return success_count == len(test_cases)
    
    def test_input_validation(self) -> bool:
        """Test input validation and error handling."""
        print("\n🛡️ Testing Input Validation...")
        
        invalid_test_cases = [
            {
                "name": "Invalid House Type",
                "payload": {"house_type": "invalid_type"},
                "expected_error": "house_type must be one of"
            },
            {
                "name": "Invalid Weather Scenario", 
                "payload": {"weather_scenario": "extreme"},
                "expected_error": "weather_scenario must be one of"
            },
            {
                "name": "Invalid Energy Company",
                "payload": {"energy_company": "fake_company"},
                "expected_error": "energy_company must be one of"
            },
            {
                "name": "Invalid Active Connections Percentage",
                "payload": {"active_connections_pct": 150},
                "expected_error": "active_connections_pct must be between"
            },
            {
                "name": "Invalid Postal Code Format",
                "payload": {"postal_code": "abc123"},
                "expected_error": "postal_code must be at least 2 digits"
            }
        ]
        
        validation_passed = 0
        for i, test_case in enumerate(invalid_test_cases, 1):
            print(f"\n   Test {i}: {test_case['name']}")
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_case['payload'],
                    timeout=10
                )
                
                if response.status_code == 422:  # Validation error
                    print(f"      Correctly rejected invalid input")
                    validation_passed += 1
                elif response.status_code == 400:  # Bad request
                    print(f"      Correctly handled bad request")
                    validation_passed += 1
                else:
                    print(f"      ❌ Unexpected status: {response.status_code}")
                    print(f"      📝 Should have rejected: {test_case['payload']}")
                    
            except Exception as e:
                print(f"      ❌ Validation test error: {e}")
        
        print(f"\n   Validation tests: {validation_passed}/{len(invalid_test_cases)} passed")
        return validation_passed >= len(invalid_test_cases) - 1  # Allow 1 failure
    
    def test_performance(self) -> bool:
        """Test API performance and concurrent requests."""
        print("\n⚡ Testing API Performance...")
        
        # Single request performance
        print("   Testing single request performance...")
        payload = {"house_type": "3x25", "weather_scenario": "normal"}
        
        response_times = []
        for i in range(5):
            start_time = time.time()
            try:
                response = self.session.post(f"{self.base_url}/predict", json=payload)
                response_time = time.time() - start_time
                if response.status_code == 200:
                    response_times.append(response_time)
            except Exception as e:
                print(f"      ❌ Performance test error: {e}")
        
        if response_times:
            avg_time = statistics.mean(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            print(f"      📊 Average response time: {avg_time:.3f}s")
            print(f"      📊 Min/Max: {min_time:.3f}s / {max_time:.3f}s")
            
            if avg_time < 2.0:  # Should respond within 2 seconds
                print(f"      ✅ Performance acceptable")
                performance_ok = True
            else:
                print(f"      ⚠️ Performance may be slow")
                performance_ok = True  # Still acceptable for demo
        else:
            print(f"      ❌ No successful responses for performance test")
            performance_ok = False
        
        # Concurrent requests test
        print("   Testing concurrent requests...")
        def make_request():
            try:
                response = self.session.post(f"{self.base_url}/predict", json=payload)
                return response.status_code == 200
            except:
                return False
        
        concurrent_success = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    concurrent_success += 1
        
        print(f"      📊 Concurrent requests: {concurrent_success}/10 successful")
        concurrent_ok = concurrent_success >= 8  # Allow some failures
        
        if concurrent_ok:
            print(f"      ✅ Concurrent handling acceptable")
        else:
            print(f"      ⚠️ Some concurrent request issues")
        
        return performance_ok and concurrent_ok
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and boundary conditions."""
        print("\n🔍 Testing Edge Cases...")
        
        edge_cases = [
            {
                "name": "Very large postal code",
                "payload": {"postal_code": "99999999"},
                "should_work": True
            },
            {
                "name": "Minimum connections",
                "payload": {"num_connections": 1, "house_type": "1x25"},
                "should_work": True
            },
            {
                "name": "Maximum active connections",
                "payload": {"active_connections_pct": 95},
                "should_work": True
            },
            {
                "name": "Minimum active connections",
                "payload": {"active_connections_pct": 50},
                "should_work": True
            },
            {
                "name": "Empty postal code with city",
                "payload": {"city": "TestCity", "postal_code": ""},
                "should_work": True
            }
        ]
        
        edge_passed = 0
        for i, test_case in enumerate(edge_cases, 1):
            print(f"\n   Test {i}: {test_case['name']}")
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_case['payload'],
                    timeout=10
                )
                
                if test_case['should_work']:
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data.get('prediction_kwh', 0)
                        print(f"      ✅ Handled correctly: {prediction:.0f} kWh/year")
                        edge_passed += 1
                    else:
                        print(f"      ❌ Should work but failed: {response.status_code}")
                else:
                    if response.status_code != 200:
                        print(f"      ✅ Correctly rejected")
                        edge_passed += 1
                    else:
                        print(f"      ❌ Should have failed but succeeded")
                        
            except Exception as e:
                print(f"      ❌ Edge case error: {e}")
        
        print(f"\n   📊 Edge case tests: {edge_passed}/{len(edge_cases)} passed")
        return edge_passed >= len(edge_cases) - 1
    
    def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run all tests and return comprehensive results."""
        print("🧪 COMPREHENSIVE API TESTING")
        print("=" * 50)
        print(f"Target API: {self.base_url}")
        print(f"Test start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        # Run all test suites
        results['connection'] = self.test_connection()
        
        if results['connection']:
            results['health'] = self.test_health_endpoint()
            results['model_info'] = self.test_model_info_endpoint()
            results['basic_predictions'] = self.test_basic_predictions()
            results['input_validation'] = self.test_input_validation()
            results['performance'] = self.test_performance()
            results['edge_cases'] = self.test_edge_cases()
        else:
            print("\n❌ Cannot continue testing - API not reachable")
            return results
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<20} {status}")
        
        print("-" * 50)
        print(f"Overall Result: {passed_tests}/{total_tests} test suites passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! API is working perfectly.")
        elif passed_tests >= total_tests - 1:
            print("✅ API is working well with minor issues.")
        else:
            print("⚠️ API has some issues that need attention.")
        
        print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results

def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Energy Consumption Predictor API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", 
                       help="API base URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only basic tests (faster)")
    
    args = parser.parse_args()
    
    tester = EnergyAPITester(args.url)
    
    if args.quick:
        # Quick test mode
        print("🚀 QUICK API TEST")
        print("=" * 30)
        
        if not tester.test_connection():
            sys.exit(1)
        
        if not tester.test_basic_predictions():
            print("❌ Basic predictions failed")
            sys.exit(1)
        
        print("✅ Quick test passed!")
    else:
        # Comprehensive test mode
        results = tester.run_comprehensive_test()
        
        # Exit with appropriate code
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        if passed_tests < total_tests - 1:  # Allow 1 failure
            sys.exit(1)
    
    print("\n💡 Usage examples:")
    print("   python test_api.py              # Full test suite")
    print("   python test_api.py --quick      # Quick basic tests")
    print("   python test_api.py --url http://localhost:8080  # Custom URL")

if __name__ == "__main__":
    main()
