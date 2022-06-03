# Introduction
This is my python implementation of "poisson surface reconstruction" [(Kazhdan et al. 2006)](https://hhoppe.com/proj/poissonrecon/) for fun.
I strictly followed [this implementation instruction](https://github.com/alecjacobson/geometry-processing-mesh-reconstruction#b-b-b-b-but-the-input-normals-might-not-be-at-grid-node-locations), [the sample C++ codes](https://github.com/j20232/geometry-processing-mesh-reconstruction/tree/master/src) (not the official implementation), and [the code explanation](https://mocobt.hatenablog.com/entry/2019/12/28/201236).

# Requirments
The code was tested using 
`python == 3.7` and packages in `requirements.txt`.

# Usage
```bibtex
python poisson_surface_reconstruction.py --path data/wheel.ply
```
The input is a point cloud with normals;
The output is a reconstructed mesh.
The sample data is taken from "shape as points" [(Peng et al. 2021)](https://github.com/autonomousvision/shape_as_points).
