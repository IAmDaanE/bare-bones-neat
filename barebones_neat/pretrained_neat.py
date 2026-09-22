from .compiled_genome import CompiledGenome
import json

class PreTrainedNEAT:
    def __init__(model_location, use_softmax):
        conns_lib = {}
        weights_lib = {}
        biases_lib = {}
        with open(f"{model_location}/conns.json", "r") as f:
            conns_lib = f.read()
        with open(f"{model_location}/weights.json", "r") as f:
            weights_lib = f.read()
        with open(f"{model_location}/biases.json", "r") as f:
            biases_lib = f.read()
        conns = conns_lib["conns"]
        weights = weights_lib["weights"]
        biases = biases_lib["biases"]
        self.compiled_genome = CompiledGenome(conns, weights, biases, use_softmax)
    
    def forward(self, inputs):
        return self.compiled_genome.forward(inputs)