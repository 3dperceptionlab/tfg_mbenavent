# Context-Based Action Estimation with YOLO

This repository contains the developments started in the Bacherlor's Degree in Computer Science of Manuel Benavent-Lledo and extended for his Master's Thesis. The advances have been also published in international conferences.

The aim of this project is to provide a context-based action estimation based on the recognition of the objects and the hands in the scene.

## Project Structure

The folders `ADL`, `EGO-DAILY` and `EPIC-KITCHENS` (we use [Epic-Kitchens 55](https://github.com/epic-kitchens/epic-kitchens-55-annotations)) contain the annotations and scripts to extract the relevant ones for those datasets, used to train YOLO models.

The `MIT_INDOOR` folder contains the annotations and code for fine-tuning and running VGG16 architecure for scene recognition.

The [`Pipeline`](Pipeline) folder contains the YOLO architecture and the action estimation architecture, further details are provided when accessing the folder.

## Docker

A docker image and launch script is provided to run this architecture.

## Citations

The following papers have been published based on the different versions of the project:

- [Bachelor's Thesis in Computer Science at the University of Alicante](http://hdl.handle.net/10045/116138)
- [Interaction Estimation in Egocentric Videos via Simultaneous Hand-Object Recognition](https://doi.org/10.1007/978-3-030-87869-6_42). Published in the 16th International Conference on Soft Computing Models in Industrial and Environmental Applications (SOCO 2021). SOCO 2021. Advances in Intelligent Systems and Computing, vol 1401.
- [Predicting Human-Object Interactions in Egocentric Videos](https://doi.org/10.1109/IJCNN55064.2022.9892910), 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-7
- [Master's Thesis in Automation and Robotics at the University of Alicante](http://hdl.handle.net/10045/125585)

```
@INPROCEEDINGS{9892910,
  author={Benavent-Lledo, Manuel and Oprea, Sergiu and Castro-Vargas, John Alejandro and Mulero-Perez, David and Garcia-Rodriguez, Jose},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Predicting Human-Object Interactions in Egocentric Videos}, 
  year={2022},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IJCNN55064.2022.9892910}}
```

## Author

Manuel Benavent-Lledo ([mbenavent@dtic.ua.es](mailto:mbenavent@dtic.ua.es))