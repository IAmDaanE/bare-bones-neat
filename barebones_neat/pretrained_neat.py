from .inference_compiled_genome import InferenceCompiledGenome
import json

class PreTrainedNEAT:
    def __init__(self, model_location, use_softmax):
        conns_lib = {}
        weights_lib = {}
        biases_lib = {}
        nodes_lib = {}
        with open(f"{model_location}/conns.json", "r") as f:
            conns_lib = f.read()
        with open(f"{model_location}/weights.json", "r") as f:
            weights_lib = f.read()
        with open(f"{model_location}/biases.json", "r") as f:
            biases_lib = f.read()
        with open(f"{model_location}/nodes.json", "r") as f:
            nodes_lib = f.read()
        conns = conns_lib["conns"]
        weights = weights_lib["weights"]
        biases = biases_lib["biases"]
        nodes = nodes_lib["nodes"]
        self.compiled_genome = InferenceCompiledGenome(conns, weights, biases, use_softmax, nodes)
    
    def forward(self, inputs):
        return self.compiled_genome.forward(inputs)