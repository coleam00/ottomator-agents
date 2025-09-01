#!/usr/bin/env python3
"""
Comprehensive performance testing script for the Medical RAG system.

This script tests:
1. API response times
2. Database query performance
3. Caching effectiveness
4. Concurrent request handling
5. Memory usage patterns
"""

import asyncio
import time
import json
import statistics
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime
import aiohttp
import psutil
import argparse
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd

# Test configuration
BASE_URL = "http://localhost:8058"
TEST_QUERIES = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "What medications are used for heart disease?",
    "Explain the causes of migraine headaches",
    "What are the risk factors for stroke?",
    "How is cancer diagnosed?",
    "What are the side effects of chemotherapy?",
    "Describe the treatment options for depression",
    "What causes anxiety disorders?",
    "How is arthritis managed?",
]

MEDICAL_ENTITIES = [
    "diabetes", "insulin", "blood pressure", "cholesterol",
    "heart disease", "cancer", "chemotherapy", "radiation",
    "depression", "anxiety", "medication", "surgery"
]


@dataclass
class TestResult:
    """Store test results for analysis."""
    operation: str
    duration: float
    success: bool
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance test report."""
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    response_times: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    test_results: List[TestResult] = field(default_factory=list)
    
    def add_result(self, result: TestResult):
        """Add a test result."""
        self.test_results.append(result)
        self.total_tests += 1
        
        if result.success:
            self.successful_tests += 1
            self.response_times.append(result.duration)
        else:
            self.failed_tests += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.response_times:
            return {"error": "No successful tests"}
        
        return {
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "failed_tests": self.failed_tests,
            "success_rate": f"{(self.successful_tests / self.total_tests * 100):.1f}%",
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) > 20 else max(self.response_times),
                "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) > 100 else max(self.response_times)
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": f"{(self.cache_hits / (self.cache_hits + self.cache_misses) * 100):.1f}%" if (self.cache_hits + self.cache_misses) > 0 else "N/A"
            },
            "resource_usage": {
                "avg_memory_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0,
                "avg_cpu_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0
            }
        }


class PerformanceTester:
    """Main performance testing class."""
    
    def __init__(self, base_url: str = BASE_URL, verbose: bool = False):
        """Initialize performance tester."""
        self.base_url = base_url
        self.verbose = verbose
        self.report = PerformanceReport()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Enter async context."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.session:
            await self.session.close()
    
    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    async def test_endpoint(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        name: str = ""
    ) -> TestResult:
        """Test a single endpoint."""
        url = f"{self.base_url}{endpoint}"
        name = name or f"{method} {endpoint}"
        
        try:
            start_time = time.time()
            
            if method == "GET":
                async with self.session.get(url) as response:
                    response_data = await response.json()
                    duration = time.time() - start_time
                    
                    # Check cache header
                    cache_status = response.headers.get("X-Cache-Status", "MISS")
                    if cache_status == "HIT":
                        self.report.cache_hits += 1
                    else:
                        self.report.cache_misses += 1
                    
                    result = TestResult(
                        operation=name,
                        duration=duration,
                        success=response.status == 200,
                        metadata={"status": response.status, "cache": cache_status}
                    )
            
            elif method == "POST":
                async with self.session.post(url, json=data) as response:
                    response_data = await response.json()
                    duration = time.time() - start_time
                    
                    # Check cache header
                    cache_status = response.headers.get("X-Cache-Status", "MISS")
                    if cache_status == "HIT":
                        self.report.cache_hits += 1
                    else:
                        self.report.cache_misses += 1
                    
                    result = TestResult(
                        operation=name,
                        duration=duration,
                        success=response.status == 200,
                        metadata={"status": response.status, "cache": cache_status}
                    )
            
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            self.log(f"✓ {name}: {duration:.3f}s (Cache: {cache_status})")
            return result
            
        except Exception as e:
            self.log(f"✗ {name}: {str(e)}")
            return TestResult(
                operation=name,
                duration=0,
                success=False,
                error=str(e)
            )
    
    async def test_health_check(self) -> TestResult:
        """Test health check endpoint."""
        return await self.test_endpoint("GET", "/health", name="Health Check")
    
    async def test_vector_search(self, query: str) -> TestResult:
        """Test vector search."""
        data = {"query": query, "limit": 10}
        return await self.test_endpoint(
            "POST", "/search/vector", data, 
            name=f"Vector Search: {query[:30]}..."
        )
    
    async def test_graph_search(self, query: str) -> TestResult:
        """Test graph search."""
        data = {"query": query}
        return await self.test_endpoint(
            "POST", "/search/graph", data,
            name=f"Graph Search: {query[:30]}..."
        )
    
    async def test_hybrid_search(self, query: str) -> TestResult:
        """Test hybrid search."""
        data = {"query": query, "limit": 10}
        return await self.test_endpoint(
            "POST", "/search/hybrid", data,
            name=f"Hybrid Search: {query[:30]}..."
        )
    
    async def test_chat(self, message: str, session_id: Optional[str] = None) -> TestResult:
        """Test chat endpoint."""
        data = {
            "message": message,
            "session_id": session_id or f"test-session-{int(time.time())}"
        }
        return await self.test_endpoint(
            "POST", "/chat", data,
            name=f"Chat: {message[:30]}..."
        )
    
    async def test_concurrent_requests(self, num_requests: int = 10) -> List[TestResult]:
        """Test concurrent request handling."""
        self.log(f"\nTesting {num_requests} concurrent requests...")
        
        tasks = []
        for i in range(num_requests):
            query = random.choice(TEST_QUERIES)
            search_type = random.choice(["vector", "graph", "hybrid"])
            
            if search_type == "vector":
                task = self.test_vector_search(query)
            elif search_type == "graph":
                task = self.test_graph_search(query)
            else:
                task = self.test_hybrid_search(query)
            
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        for result in results:
            self.report.add_result(result)
        
        return results
    
    async def test_cache_effectiveness(self) -> Dict[str, Any]:
        """Test cache effectiveness by repeating queries."""
        self.log("\nTesting cache effectiveness...")
        
        # Select a test query
        test_query = TEST_QUERIES[0]
        
        # First request (cache miss expected)
        result1 = await self.test_vector_search(test_query)
        self.report.add_result(result1)
        
        # Second request (cache hit expected)
        result2 = await self.test_vector_search(test_query)
        self.report.add_result(result2)
        
        # Third request (cache hit expected)
        result3 = await self.test_vector_search(test_query)
        self.report.add_result(result3)
        
        improvement = ((result1.duration - result3.duration) / result1.duration * 100) if result1.duration > 0 else 0
        
        return {
            "first_request": result1.duration,
            "second_request": result2.duration,
            "third_request": result3.duration,
            "improvement": f"{improvement:.1f}%",
            "cache_working": result3.duration < result1.duration * 0.5  # 50% improvement expected
        }
    
    async def test_memory_usage(self, duration: int = 30) -> List[float]:
        """Monitor memory usage during tests."""
        self.log(f"\nMonitoring resource usage for {duration} seconds...")
        
        memory_samples = []
        cpu_samples = []
        process = psutil.Process()
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Run a test
            query = random.choice(TEST_QUERIES)
            await self.test_vector_search(query)
            
            # Sample memory and CPU
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            memory_samples.append(memory_mb)
            cpu_samples.append(cpu_percent)
            
            await asyncio.sleep(1)
        
        self.report.memory_usage = memory_samples
        self.report.cpu_usage = cpu_samples
        
        return memory_samples
    
    async def run_comprehensive_test(self) -> PerformanceReport:
        """Run comprehensive performance test suite."""
        print("=" * 60)
        print("MEDICAL RAG PERFORMANCE TEST SUITE")
        print("=" * 60)
        
        # 1. Health check
        self.log("\n1. Testing system health...")
        health_result = await self.test_health_check()
        self.report.add_result(health_result)
        
        # 2. Individual endpoint tests
        self.log("\n2. Testing individual endpoints...")
        for query in TEST_QUERIES[:5]:
            result = await self.test_vector_search(query)
            self.report.add_result(result)
            
            result = await self.test_graph_search(query)
            self.report.add_result(result)
            
            result = await self.test_hybrid_search(query)
            self.report.add_result(result)
        
        # 3. Cache effectiveness
        self.log("\n3. Testing cache effectiveness...")
        cache_results = await self.test_cache_effectiveness()
        
        # 4. Concurrent requests
        await self.test_concurrent_requests(20)
        
        # 5. Chat endpoint test
        self.log("\n4. Testing chat endpoint...")
        session_id = f"perf-test-{int(time.time())}"
        for i, message in enumerate(TEST_QUERIES[:3]):
            result = await self.test_chat(message, session_id)
            self.report.add_result(result)
        
        # 6. Resource monitoring
        await self.test_memory_usage(10)
        
        return self.report
    
    def generate_report(self, output_file: str = "performance_report.json"):
        """Generate performance report."""
        summary = self.report.get_summary()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST RESULTS")
        print("=" * 60)
        
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']}")
        
        if 'response_times' in summary:
            rt = summary['response_times']
            print(f"\nResponse Times (seconds):")
            print(f"  Min: {rt['min']:.3f}")
            print(f"  Max: {rt['max']:.3f}")
            print(f"  Mean: {rt['mean']:.3f}")
            print(f"  Median: {rt['median']:.3f}")
            print(f"  P95: {rt['p95']:.3f}")
            print(f"  P99: {rt['p99']:.3f}")
        
        print(f"\nCache Performance:")
        print(f"  Hits: {summary['cache']['hits']}")
        print(f"  Misses: {summary['cache']['misses']}")
        print(f"  Hit Rate: {summary['cache']['hit_rate']}")
        
        print(f"\nResource Usage:")
        print(f"  Avg Memory: {summary['resource_usage']['avg_memory_mb']:.1f} MB")
        print(f"  Avg CPU: {summary['resource_usage']['avg_cpu_percent']:.1f}%")
        
        # Save detailed report
        detailed_report = {
            "summary": summary,
            "test_results": [
                {
                    "operation": r.operation,
                    "duration": r.duration,
                    "success": r.success,
                    "error": r.error,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata
                }
                for r in self.report.test_results
            ],
            "memory_usage": self.report.memory_usage,
            "cpu_usage": self.report.cpu_usage
        }
        
        with open(output_file, "w") as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_file}")
        
        return summary
    
    def plot_results(self, output_dir: str = "."):
        """Generate performance visualization plots."""
        if not self.report.response_times:
            print("No data to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Response time distribution
        axes[0, 0].hist(self.report.response_times, bins=20, edgecolor='black')
        axes[0, 0].set_title('Response Time Distribution')
        axes[0, 0].set_xlabel('Response Time (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Response time over time
        axes[0, 1].plot(self.report.response_times, marker='o', markersize=3)
        axes[0, 1].set_title('Response Time Over Test Run')
        axes[0, 1].set_xlabel('Test Number')
        axes[0, 1].set_ylabel('Response Time (seconds)')
        
        # 3. Memory usage
        if self.report.memory_usage:
            axes[1, 0].plot(self.report.memory_usage, color='green')
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Memory (MB)')
        
        # 4. CPU usage
        if self.report.cpu_usage:
            axes[1, 1].plot(self.report.cpu_usage, color='red')
            axes[1, 1].set_title('CPU Usage')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('CPU (%)')
        
        plt.tight_layout()
        output_file = f"{output_dir}/performance_charts_{int(time.time())}.png"
        plt.savefig(output_file)
        print(f"Performance charts saved to: {output_file}")


async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Performance testing for Medical RAG system")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the API")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite")
    parser.add_argument("--output", default="performance_report.json", help="Output file for report")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots")
    
    args = parser.parse_args()
    
    async with PerformanceTester(args.url, args.verbose) as tester:
        if args.quick:
            # Quick test
            print("Running quick performance test...")
            await tester.test_health_check()
            await tester.test_concurrent_requests(5)
            cache_results = await tester.test_cache_effectiveness()
            print(f"Cache effectiveness: {cache_results}")
        else:
            # Comprehensive test
            await tester.run_comprehensive_test()
        
        # Generate report
        tester.generate_report(args.output)
        
        # Generate plots if requested
        if args.plot:
            tester.plot_results()


if __name__ == "__main__":
    asyncio.run(main())