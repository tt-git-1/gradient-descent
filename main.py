#!/usr/bin/env python3
"""
Animation script for statistics and AI
Visualizes the gradient descent optimization process
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Font settings for Japanese display
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Noto Sans CJK JP']

# Animation settings
FPS = 15
DURATION = 30  # seconds (approximately 33 seconds)
TOTAL_FRAMES = FPS * DURATION  # 1000 frames
DATA = 50

# Output file name
OUTPUT_FILE = "gradient_descent_animation.mp4"

def create_gradient_descent_animation():
    """
    Create an animation visualizing the gradient descent optimization process
    """
    
    # Figure settings
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Graph settings (initial values are temporary as they will be dynamically changed later)
    ax.set_title('Gradient Descent Optimization', fontsize=16, fontweight='bold')
    ax.set_xlabel('Parameter (θ)', fontsize=12)
    ax.set_ylabel('Loss (L)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Loss function for gradient descent (a rugged function - complex function combining sine waves)
    def loss_function(x):
        # Complex function combining sine waves
        # Has many local minima and maxima
        return 0.5 * np.sin(3 * x) + 0.3 * np.sin(5 * x) + 0.2 * np.sin(7 * x) + 0.1 * x**2
    
    def loss_gradient(x):
        # Derivative
        # d/dx[sin(kx)] = k*cos(kx)
        return 0.5 * 3 * np.cos(3 * x) + 0.3 * 5 * np.cos(5 * x) + 0.2 * 7 * np.cos(7 * x) + 0.2 * x
    
    # Initial parameter (set randomly)
    # np.random.seed()  # Don't fix seed to get different initial values each time
    initial_theta = np.random.uniform(-DATA, DATA)  # Random value between -50 and 50
    learning_rate = 0.002  # Lower learning rate for stability
    momentum = 0.99  # Momentum coefficient (high to help escape local minima)
    
    # Data preparation
    x_range = np.linspace(-DATA, DATA, 1000)
    y_range = loss_function(x_range)
    
    # List to store animation data (passed to animate function via closure)
    theta_history = [initial_theta]
    loss_history = [loss_function(initial_theta)]
    
    # Current parameter (updated in animate function)
    current_theta = initial_theta
    # Velocity variable for momentum
    velocity = 0.0
    
    # Initial plot
    line, = ax.plot([], [], 'r-', linewidth=2, label='Loss Function')
    point, = ax.plot([], [], 'ro', markersize=12, label='Current θ')
    tangent_line, = ax.plot([], [], 'g--', linewidth=1, label='Gradient')
    ax.legend(loc='upper right')
    
    # Text display
    text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                  verticalalignment='top', fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        """Initialization function"""
        line.set_data(x_range, y_range)
        point.set_data([initial_theta], [loss_function(initial_theta)])
        tangent_line.set_data([], [])
        text.set_text(f'Iteration: 0\nθ: {initial_theta:.4f}\nLoss: {loss_function(initial_theta):.4f}\nGradient: {loss_gradient(initial_theta):.4f}')
        return [line, point, tangent_line, text]
    
    def animate(frame):
        """Animation frame update function"""
        nonlocal current_theta, velocity, theta_history, loss_history
        
        # Gradient descent update (with momentum)
        gradient = loss_gradient(current_theta)
        # Momentum update: v = momentum * v - learning_rate * gradient
        # (Negative sign needed to move in the opposite direction of the gradient)
        velocity = momentum * velocity - learning_rate * gradient
        # Parameter update: θ = θ + v (v already contains the negative sign)
        current_theta = current_theta + velocity
        
        # Limit parameter range (keep within -50 to 50)
        current_theta = np.clip(current_theta, -DATA, DATA)
        
        # Add to history
        theta_history.append(current_theta)
        loss_history.append(loss_function(current_theta))
        
        # Update graph
        point.set_data([current_theta], [loss_function(current_theta)])
        
        # Display gradient (tangent line)
        x_tangent = np.linspace(current_theta - 1, current_theta + 1, 10)
        y_tangent = loss_function(current_theta) + gradient * (x_tangent - current_theta)
        tangent_line.set_data(x_tangent, y_tangent)
        
        # Update text
        text.set_text(f'Iteration: {frame}\nθ: {current_theta:.4f}\nLoss: {loss_function(current_theta):.4f}\nGradient: {gradient:.4f}\nVelocity: {velocity:.4f}')
        
        # Dynamically change display range (center on parameter)
        margin_x = 2.0  # X-axis margin
        margin_y = 1.0  # Y-axis margin
        
        # Set display range centered on current parameter
        ax.set_xlim(current_theta - margin_x, current_theta + margin_x)
        # Set Y-axis range based on loss value (minimum as reference)
        current_loss = loss_function(current_theta)
        ax.set_ylim(current_loss - margin_y, current_loss + margin_y * 3)
        
        return [line, point, tangent_line, text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=TOTAL_FRAMES, interval=1000/FPS, blit=True, repeat=False
    )
    
    return fig, anim

def save_animation(anim):
    """Save animation"""
    print(f"Saving animation... ({OUTPUT_FILE})")
    
    # Save as MP4
    anim.save(OUTPUT_FILE, writer='ffmpeg', fps=FPS, 
              dpi=100, bitrate=1800)
    
    print(f"MP4 saved: {OUTPUT_FILE}")

def main():
    """Main function"""
    print("=" * 60)
    print("Statistics & AI Animation Generator")
    print("Gradient Descent Visualization")
    print("=" * 60)
    print()
    
    # Create animation
    print("Generating animation...")
    fig, anim = create_gradient_descent_animation()
    
    # Save
    save_animation(anim)
    
    print()
    print("Done!")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()