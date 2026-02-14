# Gradient Descent with Momentum Simulation

This project visualizes the gradient descent optimization algorithm with momentum using Python animations. It demonstrates how the algorithm navigates through a complex loss function to find optimal parameters.

This code was written by **Qwen3-Coder-Next-8bit**.

## Overview

The simulation creates an animated visualization of gradient descent optimization, showing:
- The loss function curve (a complex function combining multiple sine waves with a quadratic term)
- The current parameter position (θ) on the curve
- The gradient (tangent line) at the current position
- Real-time updates of iteration count, parameter value, loss, gradient, and velocity

![Gradient Descent Animation](example.gif)

*Animation showing gradient descent with momentum optimizing a complex loss function*

## Features

- **Momentum-based Optimization**: Uses momentum to help escape local minima and accelerate convergence
- **Dynamic Visualization**: Real-time updates of the parameter position and gradient
- **Adaptive View**: Automatically adjusts the plot range to keep the optimization process in focus
- **Customizable Parameters**: Easy to modify learning rate, momentum coefficient, and other settings

## Loss Function

The optimization targets a complex loss function with multiple local minima and maxima:

```
L(θ) = 0.5·sin(3θ) + 0.3·sin(5θ) + 0.2·sin(7θ) + 0.1·θ²
```

The gradient is computed as:

```
dL/dθ = 1.5·cos(3θ) + 1.5·cos(5θ) + 1.4·cos(7θ) + 0.2·θ
```

## Installation

This project can be installed using any of the following methods:

### Using uv (recommended)
```bash
uv sync
```

### Using pip
```bash
pip install numpy matplotlib pillow
```

### Using conda
```bash
conda install numpy matplotlib pillow
```

## Usage

Run the script to generate the animation:

```bash
python main.py
```

This will create a video file `gradient_descent_animation.mp4` in the current directory.

## Configuration

You can modify the following parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FPS` | 15 | Frames per second for the animation |
| `DURATION` | 30 | Duration in seconds |
| `DATA` | 50 | Range for parameter values (-DATA to +DATA) |
| `learning_rate` | 0.002 | Step size for each update |
| `momentum` | 0.99 | Momentum coefficient (higher = more momentum) |

## How It Works

1. **Initialization**: Starts with a random parameter value
2. **Gradient Computation**: Calculates the gradient at the current position
3. **Velocity Update**: Updates velocity using momentum: `v = momentum × v - learning_rate × gradient`
4. **Parameter Update**: Updates the parameter: `θ = θ + v`
5. **Clipping**: Constrains the parameter within the valid range
6. **Repeat**: Continues until the animation duration is reached

## Requirements

- Python 3.12+
- NumPy
- Matplotlib
- Pillow
- FFmpeg (for video encoding)

## Author

This code was written by **Qwen3-Coder-Next-8bit**.

## License

This project is for educational purposes.
