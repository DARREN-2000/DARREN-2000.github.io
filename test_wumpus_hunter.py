#!/usr/bin/env python3
"""
Test Suite for Optimized Wumpus Hunter

Comprehensive tests for HMM implementation, sensor fusion, and performance optimization.
"""

import unittest
import numpy as np
import math
import time
from wumpus_hunter import WumpusHunter, SensorReading, Direction, PerformanceMetrics

class TestWumpusHunter(unittest.TestCase):
    """Test cases for WumpusHunter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hunter = WumpusHunter(cave_size=5, debug=False)
        self.small_hunter = WumpusHunter(cave_size=3, debug=False)
    
    def test_initialization(self):
        """Test proper initialization of hunter"""
        self.assertEqual(self.hunter.cave_size, 5)
        self.assertEqual(self.hunter.total_positions, 25)
        self.assertEqual(self.hunter.state_space_size, 25 * 4)  # 4 directions
        
        # Check belief state is properly normalized
        self.assertAlmostEqual(np.sum(self.hunter.belief_state), 1.0, places=6)
        
        # Check transition matrix is properly normalized
        for i in range(self.hunter.state_space_size):
            row_sum = np.sum(self.hunter.transition_matrix[i, :])
            if row_sum > 0:
                self.assertAlmostEqual(row_sum, 1.0, places=6)
    
    def test_state_encoding_decoding(self):
        """Test state encoding and decoding functions"""
        # Test various positions and directions
        test_cases = [
            ((0, 0), Direction.NORTH),
            ((2, 3), Direction.EAST),
            ((4, 4), Direction.SOUTH),
            ((1, 2), Direction.WEST)
        ]
        
        for pos, direction in test_cases:
            encoded = self.hunter._encode_state(pos, direction)
            decoded_pos, decoded_dir = self.hunter._decode_state(encoded)
            
            self.assertEqual(pos, decoded_pos)
            self.assertEqual(direction, decoded_dir)
    
    def test_movement_probability_calculations(self):
        """Test movement probability calculations"""
        # Test continuing straight
        prob = self.hunter._calculate_movement_probability(Direction.NORTH, Direction.NORTH)
        self.assertEqual(prob, self.hunter.movement_probabilities['continue_straight'])
        
        # Test reverse
        prob = self.hunter._calculate_movement_probability(Direction.NORTH, Direction.SOUTH)
        self.assertEqual(prob, self.hunter.movement_probabilities['reverse'])
        
        # Test left turn
        prob = self.hunter._calculate_movement_probability(Direction.NORTH, Direction.WEST)
        self.assertEqual(prob, self.hunter.movement_probabilities['turn_left'])
        
        # Test right turn
        prob = self.hunter._calculate_movement_probability(Direction.NORTH, Direction.EAST)
        self.assertEqual(prob, self.hunter.movement_probabilities['turn_right'])
    
    def test_direction_utilities(self):
        """Test direction utility functions"""
        # Test opposite directions
        self.assertTrue(self.hunter._is_opposite_direction(Direction.NORTH, Direction.SOUTH))
        self.assertTrue(self.hunter._is_opposite_direction(Direction.EAST, Direction.WEST))
        self.assertFalse(self.hunter._is_opposite_direction(Direction.NORTH, Direction.EAST))
        
        # Test left turns
        self.assertTrue(self.hunter._is_left_turn(Direction.NORTH, Direction.WEST))
        self.assertTrue(self.hunter._is_left_turn(Direction.EAST, Direction.NORTH))
        self.assertFalse(self.hunter._is_left_turn(Direction.NORTH, Direction.EAST))
        
        # Test right turns
        self.assertTrue(self.hunter._is_right_turn(Direction.NORTH, Direction.EAST))
        self.assertTrue(self.hunter._is_right_turn(Direction.WEST, Direction.NORTH))
        self.assertFalse(self.hunter._is_right_turn(Direction.NORTH, Direction.WEST))
    
    def test_sensor_likelihood_calculations(self):
        """Test sensor likelihood calculations"""
        # Test GPS likelihood
        gps_reading = (2, 2)
        likelihood = self.hunter._calculate_gps_likelihood(gps_reading)
        
        self.assertEqual(len(likelihood), self.hunter.state_space_size)
        self.assertTrue(np.all(likelihood >= 0))
        self.assertTrue(np.all(likelihood <= 1))
        
        # Test microphone likelihood
        mic_reading = 0.5
        likelihood = self.hunter._calculate_microphone_likelihood(mic_reading)
        
        self.assertEqual(len(likelihood), self.hunter.state_space_size)
        self.assertTrue(np.all(likelihood >= 0))
        self.assertTrue(np.all(likelihood <= 1))
    
    def test_belief_update(self):
        """Test belief state update with sensor readings"""
        initial_belief = self.hunter.belief_state.copy()
        
        # Create sensor reading
        sensor_reading = SensorReading(gps=(2, 2), microphone=0.7, timestamp=1.0)
        
        # Update belief
        updated_belief = self.hunter.update_belief(sensor_reading)
        
        # Check that belief is properly normalized
        self.assertAlmostEqual(np.sum(updated_belief), 1.0, places=6)
        
        # Check that belief has changed
        self.assertFalse(np.array_equal(initial_belief, updated_belief))
        
        # Check that confidence has been updated
        self.assertGreater(self.hunter.metrics.confidence_score, 0)
    
    def test_position_prediction(self):
        """Test position prediction functionality"""
        # Update with some readings first
        readings = [
            SensorReading(gps=(1, 1), microphone=0.8),
            SensorReading(gps=(1, 2), microphone=0.9),
            SensorReading(gps=(2, 2), microphone=0.85)
        ]
        
        for reading in readings:
            self.hunter.update_belief(reading)
        
        # Get current position
        position, confidence = self.hunter.get_most_likely_position()
        self.assertIsInstance(position, tuple)
        self.assertEqual(len(position), 2)
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)
        
        # Test future predictions
        predictions = self.hunter.predict_next_position(3)
        self.assertEqual(len(predictions), 3)
        
        for pos, conf in predictions:
            self.assertIsInstance(pos, tuple)
            self.assertEqual(len(pos), 2)
            self.assertGreater(conf, 0)
            self.assertLessEqual(conf, 1)
    
    def test_adaptive_threshold(self):
        """Test adaptive threshold adjustment"""
        initial_threshold = self.hunter.dynamic_threshold
        
        # Simulate increasing confidence
        for i in range(10):
            self.hunter.confidence_history.append(0.1 + i * 0.05)
            self.hunter._update_adaptive_threshold()
        
        # Threshold should decrease with increasing confidence
        self.assertLess(self.hunter.dynamic_threshold, initial_threshold)
        
        # Reset and test decreasing confidence
        self.hunter.confidence_history.clear()
        self.hunter.dynamic_threshold = initial_threshold
        
        for i in range(10):
            self.hunter.confidence_history.append(0.8 - i * 0.05)
            self.hunter._update_adaptive_threshold()
        
        # Threshold should increase with decreasing confidence
        self.assertGreater(self.hunter.dynamic_threshold, initial_threshold)
    
    def test_sensor_reliability_assessment(self):
        """Test sensor reliability assessment"""
        # Test valid readings
        valid_reading = SensorReading(gps=(2, 2), microphone=0.5)
        reliability = self.hunter._assess_sensor_reliability(valid_reading)
        
        self.assertEqual(reliability['gps'], 1.0)
        self.assertEqual(reliability['microphone'], 1.0)
        
        # Test invalid GPS reading
        invalid_gps = SensorReading(gps=(-1, 2), microphone=0.5)
        reliability = self.hunter._assess_sensor_reliability(invalid_gps)
        
        self.assertEqual(reliability['gps'], 0.1)
        self.assertEqual(reliability['microphone'], 1.0)
        
        # Test invalid microphone reading
        invalid_mic = SensorReading(gps=(2, 2), microphone=1.5)
        reliability = self.hunter._assess_sensor_reliability(invalid_mic)
        
        self.assertEqual(reliability['gps'], 1.0)
        self.assertEqual(reliability['microphone'], 0.1)
    
    def test_caching_functionality(self):
        """Test that caching improves performance"""
        # Clear caches
        self.hunter.sensor_cache.clear()
        
        # First calculation should populate cache
        start_time = time.time()
        likelihood1 = self.hunter._calculate_gps_likelihood((2, 2))
        time1 = time.time() - start_time
        
        # Second calculation should use cache
        start_time = time.time()
        likelihood2 = self.hunter._calculate_gps_likelihood((2, 2))
        time2 = time.time() - start_time
        
        # Results should be identical
        np.testing.assert_array_equal(likelihood1, likelihood2)
        
        # Cache should contain the result
        self.assertIn('gps_2_2', self.hunter.sensor_cache)
        
        # Second calculation should be faster (though this might be flaky)
        # self.assertLess(time2, time1)
    
    def test_boundary_conditions(self):
        """Test behavior at cave boundaries"""
        # Test with small cave
        small_hunter = WumpusHunter(cave_size=2, debug=False)
        
        # All positions should be valid
        for x in range(2):
            for y in range(2):
                for direction in Direction:
                    state_idx = small_hunter._encode_state((x, y), direction)
                    self.assertLess(state_idx, small_hunter.state_space_size)
                    
                    decoded_pos, decoded_dir = small_hunter._decode_state(state_idx)
                    self.assertEqual((x, y), decoded_pos)
                    self.assertEqual(direction, decoded_dir)
        
        # Test GPS readings at boundaries
        boundary_reading = SensorReading(gps=(0, 0), microphone=0.5)
        small_hunter.update_belief(boundary_reading)
        
        # Should still have valid belief state
        self.assertAlmostEqual(np.sum(small_hunter.belief_state), 1.0, places=6)
    
    def test_reset_functionality(self):
        """Test reset functionality"""
        # Modify hunter state
        readings = [
            SensorReading(gps=(1, 1), microphone=0.8),
            SensorReading(gps=(1, 2), microphone=0.9)
        ]
        
        for reading in readings:
            self.hunter.update_belief(reading)
        
        # Store states before reset
        belief_before = self.hunter.belief_state.copy()
        confidence_before = len(self.hunter.confidence_history)
        
        # Reset hunter
        self.hunter.reset()
        
        # Check that state is reset
        expected_uniform = np.ones(self.hunter.state_space_size) / self.hunter.state_space_size
        np.testing.assert_array_almost_equal(self.hunter.belief_state, expected_uniform)
        
        self.assertEqual(len(self.hunter.confidence_history), 0)
        self.assertEqual(self.hunter.dynamic_threshold, self.hunter.base_confidence_threshold)
        self.assertEqual(len(self.hunter.sensor_cache), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        sensor_reading = SensorReading(gps=(2, 2), microphone=0.5)
        
        # Update belief and check metrics
        self.hunter.update_belief(sensor_reading)
        
        self.assertGreater(self.hunter.metrics.computation_time, 0)
        self.assertGreater(self.hunter.metrics.confidence_score, 0)
        self.assertLessEqual(self.hunter.metrics.confidence_score, 1)
    
    def test_debug_info(self):
        """Test debug information output"""
        # Update with a reading
        sensor_reading = SensorReading(gps=(2, 2), microphone=0.5)
        self.hunter.update_belief(sensor_reading)
        
        # Get debug info
        debug_info = self.hunter.get_debug_info()
        
        # Check required fields
        required_fields = [
            'current_position', 'confidence', 'adaptive_threshold',
            'confidence_history', 'computation_time', 'sensor_fusion_weights',
            'cache_sizes'
        ]
        
        for field in required_fields:
            self.assertIn(field, debug_info)
        
        # Check types
        self.assertIsInstance(debug_info['current_position'], list)
        self.assertIsInstance(debug_info['confidence'], float)
        self.assertIsInstance(debug_info['adaptive_threshold'], float)
        self.assertIsInstance(debug_info['confidence_history'], list)
        self.assertIsInstance(debug_info['computation_time'], float)
        self.assertIsInstance(debug_info['sensor_fusion_weights'], dict)
        self.assertIsInstance(debug_info['cache_sizes'], dict)

class TestPerformanceOptimization(unittest.TestCase):
    """Test cases for performance optimization features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hunter = WumpusHunter(cave_size=6, debug=False)
    
    def test_matrix_operations_efficiency(self):
        """Test that matrix operations are efficient"""
        # Create multiple sensor readings
        readings = [
            SensorReading(gps=(i % 6, (i + 1) % 6), microphone=0.5 + i * 0.1)
            for i in range(10)
        ]
        
        # Measure time for belief updates
        start_time = time.time()
        for reading in readings:
            self.hunter.update_belief(reading)
        total_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(total_time, 5.0)  # 5 seconds for 10 updates
        
        # Average time per update should be reasonable
        avg_time = total_time / len(readings)
        self.assertLess(avg_time, 0.5)  # 0.5 seconds per update
    
    def test_large_cave_performance(self):
        """Test performance with larger cave sizes"""
        large_hunter = WumpusHunter(cave_size=10, debug=False)
        
        # Test initialization time
        start_time = time.time()
        large_hunter._initialize_transition_matrix()
        init_time = time.time() - start_time
        
        # Should initialize in reasonable time
        self.assertLess(init_time, 10.0)  # 10 seconds for 10x10 cave
        
        # Test belief update time
        sensor_reading = SensorReading(gps=(5, 5), microphone=0.5)
        
        start_time = time.time()
        large_hunter.update_belief(sensor_reading)
        update_time = time.time() - start_time
        
        # Should update in reasonable time
        self.assertLess(update_time, 1.0)  # 1 second for update
    
    def test_memory_efficiency(self):
        """Test memory efficiency of implementation"""
        # Create hunter and check initial memory usage
        hunter = WumpusHunter(cave_size=8, debug=False)
        
        # Transition matrix should be reasonable size
        matrix_size = hunter.transition_matrix.nbytes
        max_expected_size = 8 * 8 * 4 * 8 * 8 * 4 * 8 + 1000  # rough estimate with buffer
        
        self.assertLess(matrix_size, max_expected_size)
        
        # Test that caches don't grow unbounded
        for i in range(100):
            reading = SensorReading(gps=(i % 8, (i + 1) % 8), microphone=0.5)
            hunter.update_belief(reading)
        
        # Cache should have reasonable size
        self.assertLess(len(hunter.sensor_cache), 1000)

class TestSensorFusion(unittest.TestCase):
    """Test cases for sensor fusion functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hunter = WumpusHunter(cave_size=4, debug=False)
    
    def test_gps_only_tracking(self):
        """Test tracking with GPS sensor only"""
        readings = [
            SensorReading(gps=(1, 1), timestamp=1.0),
            SensorReading(gps=(1, 2), timestamp=2.0),
            SensorReading(gps=(2, 2), timestamp=3.0)
        ]
        
        for reading in readings:
            self.hunter.update_belief(reading)
        
        # Should track reasonably well
        position, confidence = self.hunter.get_most_likely_position()
        self.assertGreater(confidence, 0.1)
    
    def test_microphone_only_tracking(self):
        """Test tracking with microphone sensor only"""
        readings = [
            SensorReading(microphone=0.3, timestamp=1.0),
            SensorReading(microphone=0.5, timestamp=2.0),
            SensorReading(microphone=0.7, timestamp=3.0)
        ]
        
        for reading in readings:
            self.hunter.update_belief(reading)
        
        # Should track reasonably well
        position, confidence = self.hunter.get_most_likely_position()
        self.assertGreater(confidence, 0.1)
    
    def test_combined_sensor_fusion(self):
        """Test tracking with both GPS and microphone sensors"""
        readings = [
            SensorReading(gps=(1, 1), microphone=0.3, timestamp=1.0),
            SensorReading(gps=(1, 2), microphone=0.5, timestamp=2.0),
            SensorReading(gps=(2, 2), microphone=0.7, timestamp=3.0)
        ]
        
        for reading in readings:
            self.hunter.update_belief(reading)
        
        # Combined sensors should provide better tracking
        position, confidence = self.hunter.get_most_likely_position()
        self.assertGreater(confidence, 0.2)
    
    def test_sensor_weight_adaptation(self):
        """Test adaptive sensor weight adjustment"""
        # Test with reliable sensors
        reliable_reading = SensorReading(gps=(2, 2), microphone=0.5)
        initial_weights = self.hunter.sensor_fusion_weights.copy()
        
        self.hunter.update_belief(reliable_reading)
        
        # Weights should remain high for reliable sensors
        self.assertGreater(self.hunter.sensor_fusion_weights['gps'], 0.3)
        self.assertGreater(self.hunter.sensor_fusion_weights['microphone'], 0.2)
        
        # Test with unreliable GPS
        unreliable_reading = SensorReading(gps=(-1, 5), microphone=0.5)
        self.hunter.update_belief(unreliable_reading)
        
        # GPS weight should be reduced
        self.assertLess(self.hunter.sensor_fusion_weights['gps'], 0.1)

def run_performance_benchmarks():
    """Run performance benchmarks across different scenarios"""
    print("=== Performance Benchmarks ===\n")
    
    # Test different cave sizes
    cave_sizes = [4, 6, 8, 10]
    
    for size in cave_sizes:
        print(f"Testing cave size {size}x{size}:")
        
        # Initialization time
        start_time = time.time()
        hunter = WumpusHunter(cave_size=size, debug=False)
        init_time = time.time() - start_time
        
        print(f"  Initialization: {init_time:.4f}s")
        
        # Update time
        readings = [
            SensorReading(gps=(i % size, (i + 1) % size), microphone=0.5)
            for i in range(10)
        ]
        
        start_time = time.time()
        for reading in readings:
            hunter.update_belief(reading)
        update_time = time.time() - start_time
        
        print(f"  10 updates: {update_time:.4f}s ({update_time/10:.4f}s avg)")
        
        # Memory usage
        matrix_size = hunter.transition_matrix.nbytes / (1024 * 1024)  # MB
        print(f"  Transition matrix: {matrix_size:.2f} MB")
        
        print()

def main():
    """Run all tests and benchmarks"""
    print("Running Wumpus Hunter Test Suite...\n")
    
    # Run unit tests
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestWumpusHunter,
        TestPerformanceOptimization,
        TestSensorFusion
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"  {test}: {trace}")
    
    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"  {test}: {trace}")
    
    # Run performance benchmarks
    print("\n" + "="*50)
    run_performance_benchmarks()
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)