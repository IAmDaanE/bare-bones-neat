# Custom NEAT Library 

My own homemade barebones_neat library (Neuro Evolution of Augmented Topologies) for training AI models.

I used this library in some of my projects:
- [a simple ai steering model](https://github.com/IAmDaanE/ai-simple-steering)

---

## About the Project

It has very nice visualizations for training and inference, made with pygame-ce

The weights and biases are just stored as regular python array, since the networks are usually not very big this doesn't slow things down too much. When training is done the model can be saved. The connections, weights, biases... Will be stored in json files. They can then later be loaded in for inference.

## Getting Started

### Installation

**Requires:** Python 3.10 - 3.14  
Run this command (preferably using a virtual environment):

```
pip install git+https://github.com/IAmDaanE/bare-bones-neat.git@master
```

### Usage

Importing
```python
import barebones_neat as bbn
```

Initiating the system
```python
neat = bbn.NEAT(population_size=30, input_size=3, output_size=4)
```

Doing a forward pass
```python
for specie in compiled_genomes:
    for compiled_genome in specie:
        outputs = compiled_genome.forward(inputs)
```

Creating a new generation based on fitness scores
```python
neat.create_new_generation()
neat.sort_into_species()
```

## License

This project is open-source and available under the MIT License.