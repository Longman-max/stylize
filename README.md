# stylize

**Regressor-based image stylization using machine learning**

A Python library that creates artistic image effects using regression trees to partition image features and learn color associations.

## Credits

Original concept and implementation by **[Alec Radford](http://github.com/newmu)**. My version includes performance optimizations with reduced npxs values for faster model training.

## Examples

Our test subject: Bella my dog, showcasing different stylization effects.

| Original | Default stylization `render(image)` |
|----------|--------------------------------------|
| ![Bella](/resources/bella-small.jpg?raw=true "Bella looking confused") | ![Bella](/example_images/more_detail.png?raw=true "Default stylization with stylize") |

### Style Variations

| Abstract `render(image, depth=4)` | Smooth `render(image, iterations=25)` |
|-----------------------------------|---------------------------------------|
| ![Bella](/example_images/abstract.png?raw=true "Abstract Bella") | ![Bella](/example_images/smoother.png?raw=true "Smooth Bella") |

| More Detail `render(image, ratio=0.00005)` | Less Detail `render(image, ratio=0.001)` |
|---------------------------------------------|-------------------------------------------|
| ![Bella](/example_images/more_detail.png?raw=true "Bella in all her glory") | ![Bella](/example_images/less_detail.png?raw=true "Bella in low fidelity") |

## How It Works

| Visualization | Explanation |
|---------------|-------------|
| ![Longman-max](/resources/longman.gif?raw=true "Visualizing how it works") | **stylize** uses regression trees and random forest regressors to create artistic effects. The algorithm works by recursively partitioning the input feature space (image pixels) and learning color associations for each partition. At each iteration, the model splits every partition in half until reaching the minimum partition size. This creates the distinctive blocky, artistic effect you see in the output. Click the gif for a larger view! |

## Quick Start

### Installation
```bash
pip install scipy numpy scikit-learn pillow
```

### Basic Usage
```python
from stylize import render
from scipy.misc import imread

image = imread('resources/bella.jpg')
defaults = render(image)
```

For more detailed examples, see `example.py`.

## Parameters

The `render()` function accepts several parameters to control the stylization:

- **`depth`**: Controls abstraction level (higher = more abstract)
- **`iterations`**: Number of smoothing passes (higher = smoother result)  
- **`ratio`**: Detail level (lower = more detail, higher = less detail)

## Performance Notes

This version includes optimizations for faster training:
- Reduced npxs (number of pixel samples) values for quicker model convergence
- Maintained visual quality while improving processing speed

## Technical Details

The stylization process leverages:
- **Regression Trees**: For learning pixel-to-color mappings
- **Random Forest Ensembles**: For improved stability and quality
- **Recursive Partitioning**: Creates the characteristic artistic blocks
- **Feature Space Mapping**: Transforms spatial coordinates to color values

## Requirements

- Python 3.x
- NumPy
- SciPy  
- scikit-learn
- PIL/Pillow

---

*Original algorithm by Alec Radford. Performance optimizations and documentation enhancements included in this version.*