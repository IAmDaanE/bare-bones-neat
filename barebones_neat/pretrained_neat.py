from .inference_compiled_genome import InferenceCompiledGenome
import json

class PreTrainedNEAT:
    def __init__(self, model_location):
        raw_conns_lib = {}
        raw_weights_lib = {}
        biases_lib = {}
        nodes_lib = {}
        details_lib = {}
        with open(model_location / "conns.json", "r") as f:
            raw_conns_lib = json.load(f)
        with open(model_location / "weights.json", "r") as f:
            raw_weights_lib = json.load(f)
        with open(model_location / "biases.json", "r") as f:
            biases_lib = json.load(f)
        with open(model_location / "nodes.json", "r") as f:
            nodes_lib = json.load(f)
        with open(model_location / "details.json", "r") as f:
            details_lib = json.load(f)
        raw_conns = raw_conns_lib["conns"]
        conns = {}
        for key, value in raw_conns.items():
            splitted = key.split(",")
            conns[(splitted[0], splitted[1])] = value
        raw_weights = raw_weights_lib["weights"]
        weights = {}
        for key, value in raw_weights.items():
            splitted = key.split(",")
            weights[(splitted[0], splitted[1])] = value
        biases = biases_lib["biases"]
        nodes = nodes_lib["nodes"]
        use_softmax = details_lib["use_softmax"]
        self.compiled_genome = InferenceCompiledGenome(conns, weights, biases, use_softmax, nodes)
    
    def forward(self, inputs):
        return self.compiled_genome.forward(inputs)