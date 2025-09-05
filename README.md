stylize
=======

Regressor based image stylization.


Usage (example.py has more detail): 
```
from stylize import render
from scipy.misc import imread

image = imread('resources/longman.jpg')
defaults = render(image)
```

Our Test Subject, my dog bella | Default stylization `render(image)`
------------- | -------------
![Bella](/resources/bella.jpg?raw=true "Bella looking confused")  | ![Bella](/example_images/more_detail.png?raw=true "Default stylization with stylize")

Abstract `render(image,depth=4)` | Smooth `render(image,iterations=25)`
------------- | -------------
![Bella](/example_images/abstract.png?raw=true "Abstract Bella")  | ![Bella](/example_images/smoother.png?raw=true "Smooth Bella")

More Detail `render(image,ratio=0.00005)` | Less Detail `render(image,ratio=0.001)`
------------- | -------------
![Bella](/example_images/more_detail.png?raw=true "Bella in all her glory")  | ![Bella](/example_images/less_detail.png?raw=true "Bella in low fidelity")
