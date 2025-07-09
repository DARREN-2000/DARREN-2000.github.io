#!/usr/bin/env python3
"""
Environment Performance Comparison for Wumpus Hunter

This script tests the optimized Wumpus Hunter across different environments
and sensor configurations to validate performance targets.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple
from wumpus_hunter import WumpusHunter, SensorReading, Direction

class EnvironmentTester:
    """Test Wumpus Hunter performance across different environments"""
    
    def __init__(self, cave_size: int = 10, num_trials: int = 100):
        self.cave_size = cave_size
        self.num_trials = num_trials
        self.results = {}
    
    def test_tracker_only(self) -> Dict:
        """Test with perfect tracking (baseline)"""
        print("Testing Tracker Only environment...")
        
        scores = []
        times = []
        
        for trial in range(self.num_trials):
            hunter = WumpusHunter(cave_size=self.cave_size, debug=False)
            
            # Generate true trajectory
            true_positions = self._generate_trajectory(20)
            
            # Perfect tracking - no sensor noise
            total_score = 0
            start_time = time.time()
            
            for i, pos in enumerate(true_positions):
                # Perfect sensor reading
                reading = SensorReading(gps=pos, microphone=1.0, timestamp=i)
                hunter.update_belief(reading)
                
                # Score based on prediction accuracy
                pred_pos, confidence = hunter.get_most_likely_position()
                distance = np.sqrt((pred_pos[0] - pos[0])**2 + (pred_pos[1] - pos[1])**2)
                score = max(0, 1.0 - distance / 5.0)  # Normalized score
                total_score += score
            
            times.append(time.time() - start_time)
            scores.append(total_score / len(true_positions))
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_time = np.mean(times)
        
        result = {
            'environment': 'tracker_only',
            'avg_score': float(avg_score),
            'std_score': float(std_score),
            'avg_time': float(avg_time),
            'target_score': 0.65,
            'passes_target': bool(avg_score >= 0.65)
        }
        
        print(f"  Average Score: {avg_score:.3f} (±{std_score:.3f})")
        print(f"  Target: 0.65 - {'PASS' if result['passes_target'] else 'FAIL'}")
        print(f"  Average Time: {avg_time:.4f}s")
        
        return result
    
    def test_gps_only(self) -> Dict:
        """Test with GPS sensor only"""
        print("Testing GPS Only environment...")
        
        scores = []
        times = []
        
        for trial in range(self.num_trials):
            hunter = WumpusHunter(cave_size=self.cave_size, debug=False)
            
            # Generate true trajectory
            true_positions = self._generate_trajectory(20)
            
            total_score = 0
            start_time = time.time()
            
            for i, pos in enumerate(true_positions):
                # GPS with noise
                noisy_gps = self._add_gps_noise(pos)
                reading = SensorReading(gps=noisy_gps, timestamp=i)
                hunter.update_belief(reading)
                
                # Score based on prediction accuracy
                pred_pos, confidence = hunter.get_most_likely_position()
                distance = np.sqrt((pred_pos[0] - pos[0])**2 + (pred_pos[1] - pos[1])**2)
                score = max(0, 1.0 - distance / 5.0)
                total_score += score
            
            times.append(time.time() - start_time)
            scores.append(total_score / len(true_positions))
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_time = np.mean(times)
        
        result = {
            'environment': 'gps_only',
            'avg_score': float(avg_score),
            'std_score': float(std_score),
            'avg_time': float(avg_time),
            'target_score': 0.35,
            'passes_target': bool(avg_score >= 0.35)
        }
        
        print(f"  Average Score: {avg_score:.3f} (±{std_score:.3f})")
        print(f"  Target: 0.35 - {'PASS' if result['passes_target'] else 'FAIL'}")
        print(f"  Average Time: {avg_time:.4f}s")
        
        return result
    
    def test_microphones_only(self) -> Dict:
        """Test with microphone sensors only"""
        print("Testing Microphones Only environment...")
        
        scores = []
        times = []
        
        for trial in range(self.num_trials):
            hunter = WumpusHunter(cave_size=self.cave_size, debug=False)
            
            # Generate true trajectory
            true_positions = self._generate_trajectory(20)
            
            total_score = 0
            start_time = time.time()
            
            for i, pos in enumerate(true_positions):
                # Microphone signal with noise
                mic_signal = self._calculate_microphone_signal(pos) + np.random.normal(0, 0.1)
                mic_signal = np.clip(mic_signal, 0, 1)
                
                reading = SensorReading(microphone=mic_signal, timestamp=i)
                hunter.update_belief(reading)
                
                # Score based on prediction accuracy
                pred_pos, confidence = hunter.get_most_likely_position()
                distance = np.sqrt((pred_pos[0] - pos[0])**2 + (pred_pos[1] - pos[1])**2)
                score = max(0, 1.0 - distance / 5.0)
                total_score += score
            
            times.append(time.time() - start_time)
            scores.append(total_score / len(true_positions))
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_time = np.mean(times)
        
        result = {
            'environment': 'microphones_only',
            'avg_score': float(avg_score),
            'std_score': float(std_score),
            'avg_time': float(avg_time),
            'target_score': 0.5,
            'passes_target': bool(avg_score >= 0.5)
        }
        
        print(f"  Average Score: {avg_score:.3f} (±{std_score:.3f})")
        print(f"  Target: 0.50 - {'PASS' if result['passes_target'] else 'FAIL'}")
        print(f"  Average Time: {avg_time:.4f}s")
        
        return result
    
    def test_microphones_no_noise(self) -> Dict:
        """Test with microphone sensors without noise"""
        print("Testing Microphones (No Noise) environment...")
        
        scores = []
        times = []
        
        for trial in range(self.num_trials):
            hunter = WumpusHunter(cave_size=self.cave_size, debug=False)
            
            # Generate true trajectory
            true_positions = self._generate_trajectory(20)
            
            total_score = 0
            start_time = time.time()
            
            for i, pos in enumerate(true_positions):
                # Clean microphone signal
                mic_signal = self._calculate_microphone_signal(pos)
                
                reading = SensorReading(microphone=mic_signal, timestamp=i)
                hunter.update_belief(reading)
                
                # Score based on prediction accuracy
                pred_pos, confidence = hunter.get_most_likely_position()
                distance = np.sqrt((pred_pos[0] - pos[0])**2 + (pred_pos[1] - pos[1])**2)
                score = max(0, 1.0 - distance / 5.0)
                total_score += score
            
            times.append(time.time() - start_time)
            scores.append(total_score / len(true_positions))
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_time = np.mean(times)
        
        result = {
            'environment': 'microphones_no_noise',
            'avg_score': float(avg_score),
            'std_score': float(std_score),
            'avg_time': float(avg_time),
            'target_score': 0.45,
            'passes_target': bool(avg_score >= 0.45)
        }
        
        print(f"  Average Score: {avg_score:.3f} (±{std_score:.3f})")
        print(f"  Target: 0.45 - {'PASS' if result['passes_target'] else 'FAIL'}")
        print(f"  Average Time: {avg_time:.4f}s")
        
        return result
    
    def test_gps_microphones(self) -> Dict:
        """Test with both GPS and microphone sensors"""
        print("Testing GPS + Microphones environment...")
        
        scores = []
        times = []
        
        for trial in range(self.num_trials):
            hunter = WumpusHunter(cave_size=self.cave_size, debug=False)
            
            # Generate true trajectory
            true_positions = self._generate_trajectory(20)
            
            total_score = 0
            start_time = time.time()
            
            for i, pos in enumerate(true_positions):
                # GPS with noise
                noisy_gps = self._add_gps_noise(pos)
                
                # Microphone signal with noise
                mic_signal = self._calculate_microphone_signal(pos) + np.random.normal(0, 0.1)
                mic_signal = np.clip(mic_signal, 0, 1)
                
                reading = SensorReading(gps=noisy_gps, microphone=mic_signal, timestamp=i)
                hunter.update_belief(reading)
                
                # Score based on prediction accuracy
                pred_pos, confidence = hunter.get_most_likely_position()
                distance = np.sqrt((pred_pos[0] - pos[0])**2 + (pred_pos[1] - pos[1])**2)
                score = max(0, 1.0 - distance / 5.0)
                total_score += score
            
            times.append(time.time() - start_time)
            scores.append(total_score / len(true_positions))
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_time = np.mean(times)
        
        result = {
            'environment': 'gps_microphones',
            'avg_score': float(avg_score),
            'std_score': float(std_score),
            'avg_time': float(avg_time),
            'target_score': 0.5,
            'passes_target': bool(avg_score >= 0.5)
        }
        
        print(f"  Average Score: {avg_score:.3f} (±{std_score:.3f})")
        print(f"  Target: 0.50 - {'PASS' if result['passes_target'] else 'FAIL'}")
        print(f"  Average Time: {avg_time:.4f}s")
        
        return result
    
    def _generate_trajectory(self, steps: int) -> List[Tuple[int, int]]:
        """Generate a realistic wumpus trajectory"""
        positions = []
        current_pos = (self.cave_size // 2, self.cave_size // 2)
        current_dir = Direction.NORTH
        
        positions.append(current_pos)
        
        for _ in range(steps - 1):
            # Movement probabilities based on our model
            if np.random.random() < 0.7:  # Continue straight
                new_dir = current_dir
            elif np.random.random() < 0.3:  # Turn
                turns = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
                new_dir = np.random.choice(turns)
            else:  # Random
                new_dir = np.random.choice(list(Direction))
            
            # Calculate new position
            new_x = current_pos[0] + new_dir.dx
            new_y = current_pos[1] + new_dir.dy
            
            # Check boundaries
            if 0 <= new_x < self.cave_size and 0 <= new_y < self.cave_size:
                current_pos = (new_x, new_y)
                current_dir = new_dir
            
            positions.append(current_pos)
        
        return positions
    
    def _add_gps_noise(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Add noise to GPS reading"""
        x, y = position
        
        # Add random noise
        noise_x = np.random.normal(0, 0.5)
        noise_y = np.random.normal(0, 0.5)
        
        noisy_x = int(np.clip(x + noise_x, 0, self.cave_size - 1))
        noisy_y = int(np.clip(y + noise_y, 0, self.cave_size - 1))
        
        return (noisy_x, noisy_y)
    
    def _calculate_microphone_signal(self, position: Tuple[int, int]) -> float:
        """Calculate microphone signal strength"""
        x, y = position
        center_x, center_y = self.cave_size // 2, self.cave_size // 2
        
        # Distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(2) * self.cave_size / 2
        
        # Signal strength decreases with distance
        signal = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, signal))
    
    def run_all_tests(self) -> Dict:
        """Run all environment tests"""
        print("=== Wumpus Hunter Environment Performance Tests ===\n")
        
        self.results['tracker_only'] = self.test_tracker_only()
        print()
        
        self.results['gps_only'] = self.test_gps_only()
        print()
        
        self.results['microphones_only'] = self.test_microphones_only()
        print()
        
        self.results['microphones_no_noise'] = self.test_microphones_no_noise()
        print()
        
        self.results['gps_microphones'] = self.test_gps_microphones()
        print()
        
        return self.results
    
    def print_summary(self):
        """Print performance summary"""
        print("=== Performance Summary ===")
        print(f"{'Environment':<20} {'Score':<8} {'Target':<8} {'Status':<8} {'Time':<8}")
        print("-" * 60)
        
        for env_name, result in self.results.items():
            status = "PASS" if result['passes_target'] else "FAIL"
            print(f"{result['environment']:<20} {result['avg_score']:<8.3f} {result['target_score']:<8.3f} {status:<8} {result['avg_time']:<8.4f}")
        
        # Overall performance
        total_passed = sum(1 for r in self.results.values() if r['passes_target'])
        total_tests = len(self.results)
        
        print(f"\nOverall Performance: {total_passed}/{total_tests} tests passed")
        
        # Expected performance targets
        expected_scores = {
            'tracker_only': 0.65,
            'gps_only': 0.35,
            'microphones_only': 0.5,
            'microphones_no_noise': 0.45,
            'gps_microphones': 0.5
        }
        
        print("\nTarget Analysis:")
        for env_name, result in self.results.items():
            expected = expected_scores.get(env_name, 0.5)
            actual = result['avg_score']
            improvement = ((actual - expected) / expected) * 100
            print(f"  {env_name}: {improvement:+.1f}% vs target")

def main():
    """Run environment performance tests"""
    tester = EnvironmentTester(cave_size=8, num_trials=20)  # Reduced for faster testing
    
    results = tester.run_all_tests()
    tester.print_summary()
    
    # Save results to file
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to performance_results.json")

if __name__ == "__main__":
    main()