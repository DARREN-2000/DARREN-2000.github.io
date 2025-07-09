#!/usr/bin/env python3
"""
Wumpus Hunter Demo Script for Portfolio Integration

This script demonstrates the optimized Wumpus Hunter implementation
for inclusion in the portfolio website.
"""

import json
from wumpus_hunter import WumpusHunter, SensorReading

def create_demo_html():
    """Create HTML demonstration for portfolio integration"""
    
    # Initialize hunter
    hunter = WumpusHunter(cave_size=6, debug=False)
    
    # Simulate a realistic tracking scenario
    scenario = {
        "title": "Wumpus Hunter - Advanced HMM Implementation",
        "description": "Real-time demonstration of probabilistic tracking using Hidden Markov Models",
        "steps": []
    }
    
    # Define sensor sequence
    sensor_sequence = [
        SensorReading(gps=(2, 2), microphone=0.4, timestamp=1.0),
        SensorReading(gps=(2, 3), microphone=0.5, timestamp=2.0),
        SensorReading(gps=(3, 3), microphone=0.6, timestamp=3.0),
        SensorReading(gps=(3, 4), microphone=0.7, timestamp=4.0),
        SensorReading(gps=(4, 4), microphone=0.8, timestamp=5.0),
        SensorReading(gps=(4, 5), microphone=0.75, timestamp=6.0),
    ]
    
    # Process each sensor reading
    for i, reading in enumerate(sensor_sequence):
        hunter.update_belief(reading)
        
        # Get current state
        position, confidence = hunter.get_most_likely_position()
        predictions = hunter.predict_next_position(2)
        should_act = hunter.should_take_action()
        
        # Format step data
        step = {
            "timestep": i + 1,
            "sensor_input": {
                "gps": reading.gps,
                "microphone": reading.microphone
            },
            "belief_state": {
                "most_likely_position": [int(position[0]), int(position[1])],
                "confidence": round(confidence, 3),
                "should_take_action": should_act
            },
            "predictions": [
                {
                    "position": [int(pos[0]), int(pos[1])],
                    "confidence": round(conf, 3)
                }
                for pos, conf in predictions
            ],
            "metrics": {
                "computation_time": round(hunter.metrics.computation_time * 1000, 2),  # ms
                "adaptive_threshold": round(hunter.dynamic_threshold, 3)
            }
        }
        
        scenario["steps"].append(step)
    
    # Final performance summary
    debug_info = hunter.get_debug_info()
    scenario["summary"] = {
        "final_confidence": debug_info["confidence"],
        "total_steps": len(sensor_sequence),
        "average_computation_time": round(sum(step["metrics"]["computation_time"] for step in scenario["steps"]) / len(scenario["steps"]), 2),
        "performance_metrics": {
            "sensor_fusion_weights": debug_info["sensor_fusion_weights"],
            "cache_efficiency": debug_info["cache_sizes"]
        }
    }
    
    return scenario

def generate_portfolio_integration():
    """Generate HTML/JSON for portfolio integration"""
    
    demo_data = create_demo_html()
    
    # Save demo data as JSON
    with open('wumpus_demo_data.json', 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    # Create HTML snippet for portfolio integration
    html_snippet = f"""
    <!-- Wumpus Hunter Project Section -->
    <div class="project-card wumpus-hunter" data-aos="fade-up">
        <div class="project-image">
            <div class="project-overlay">
                <div class="project-links">
                    <a href="wumpus_hunter.py" class="project-link" target="_blank">
                        <i class="fab fa-python"></i>
                    </a>
                    <a href="WUMPUS_HUNTER_README.md" class="project-link" target="_blank">
                        <i class="fas fa-book"></i>
                    </a>
                    <a href="#" class="project-link demo-link" onclick="showWumpusDemo()">
                        <i class="fas fa-play"></i>
                    </a>
                </div>
            </div>
            <div class="project-category">Artificial Intelligence</div>
        </div>
        <div class="project-content">
            <h3>Wumpus Hunter - Advanced HMM Implementation</h3>
            <p>Optimized probabilistic tracking system using Hidden Markov Models with sophisticated sensor fusion and adaptive decision making for real-time target tracking.</p>
            <div class="project-tech">
                <span class="tech-tag">Hidden Markov Models</span>
                <span class="tech-tag">Sensor Fusion</span>
                <span class="tech-tag">Probabilistic Tracking</span>
                <span class="tech-tag">NumPy</span>
                <span class="tech-tag">Matrix Operations</span>
            </div>
            <div class="project-metrics">
                <div class="metric">
                    <span class="metric-value">3/5</span>
                    <span class="metric-label">Environments Passed</span>
                </div>
                <div class="metric">
                    <span class="metric-value">&lt;1ms</span>
                    <span class="metric-label">Update Time</span>
                </div>
                <div class="metric">
                    <span class="metric-value">83.5%</span>
                    <span class="metric-label">Best Performance</span>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Save HTML snippet
    with open('wumpus_hunter_portfolio.html', 'w') as f:
        f.write(html_snippet)
    
    # Create JavaScript for interactive demo
    js_demo = """
    function showWumpusDemo() {
        // Load demo data
        fetch('wumpus_demo_data.json')
            .then(response => response.json())
            .then(data => {
                // Create demo modal or section
                const demoModal = document.createElement('div');
                demoModal.className = 'wumpus-demo-modal';
                demoModal.innerHTML = `
                    <div class="demo-content">
                        <h3>${data.title}</h3>
                        <p>${data.description}</p>
                        <div class="demo-steps">
                            ${data.steps.map(step => `
                                <div class="demo-step">
                                    <h4>Step ${step.timestep}</h4>
                                    <div class="sensor-data">
                                        <strong>GPS:</strong> (${step.sensor_input.gps[0]}, ${step.sensor_input.gps[1]})
                                        <strong>Microphone:</strong> ${step.sensor_input.microphone}
                                    </div>
                                    <div class="belief-state">
                                        <strong>Position:</strong> (${step.belief_state.most_likely_position[0]}, ${step.belief_state.most_likely_position[1]})
                                        <strong>Confidence:</strong> ${step.belief_state.confidence}
                                        <strong>Action:</strong> ${step.belief_state.should_take_action ? 'TAKE ACTION' : 'WAIT'}
                                    </div>
                                    <div class="predictions">
                                        <strong>Predictions:</strong>
                                        ${step.predictions.map(pred => `(${pred.position[0]}, ${pred.position[1]}) - ${pred.confidence}`).join(', ')}
                                    </div>
                                    <div class="metrics">
                                        <strong>Computation Time:</strong> ${step.metrics.computation_time}ms
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                        <div class="demo-summary">
                            <h4>Performance Summary</h4>
                            <p><strong>Final Confidence:</strong> ${data.summary.final_confidence.toFixed(3)}</p>
                            <p><strong>Average Computation Time:</strong> ${data.summary.average_computation_time}ms</p>
                            <p><strong>GPS Weight:</strong> ${data.summary.performance_metrics.sensor_fusion_weights.gps}</p>
                            <p><strong>Microphone Weight:</strong> ${data.summary.performance_metrics.sensor_fusion_weights.microphone}</p>
                        </div>
                        <button onclick="closeDemoModal()" class="close-demo">Close</button>
                    </div>
                `;
                document.body.appendChild(demoModal);
            });
    }
    
    function closeDemoModal() {
        const modal = document.querySelector('.wumpus-demo-modal');
        if (modal) {
            modal.remove();
        }
    }
    """
    
    # Save JavaScript
    with open('wumpus_demo.js', 'w') as f:
        f.write(js_demo)
    
    # Create CSS for demo styling
    css_demo = """
    .wumpus-demo-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    
    .demo-content {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
        color: #333;
    }
    
    .demo-step {
        border: 1px solid #ddd;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .sensor-data, .belief-state, .predictions, .metrics {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .close-demo {
        background: #4ECDC4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 1rem;
    }
    
    .close-demo:hover {
        background: #45B7AA;
    }
    """
    
    # Save CSS
    with open('wumpus_demo.css', 'w') as f:
        f.write(css_demo)
    
    print("Portfolio integration files created:")
    print("- wumpus_demo_data.json: Demo data")
    print("- wumpus_hunter_portfolio.html: HTML snippet")
    print("- wumpus_demo.js: Interactive JavaScript")
    print("- wumpus_demo.css: Demo styling")

def main():
    """Main function"""
    print("=== Wumpus Hunter Portfolio Integration ===\\n")
    
    # Generate portfolio integration files
    generate_portfolio_integration()
    
    print("\\n=== Demo Data Generated ===")
    
    # Show summary
    with open('wumpus_demo_data.json', 'r') as f:
        demo_data = json.load(f)
    
    print(f"Title: {demo_data['title']}")
    print(f"Total Steps: {demo_data['summary']['total_steps']}")
    print(f"Final Confidence: {demo_data['summary']['final_confidence']:.3f}")
    print(f"Average Computation Time: {demo_data['summary']['average_computation_time']}ms")
    
    print("\\nReady for portfolio integration!")

if __name__ == "__main__":
    main()