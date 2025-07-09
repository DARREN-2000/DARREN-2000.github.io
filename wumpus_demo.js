
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
    