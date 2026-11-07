import math

class InferenceCompiledGenome:
    def __init__(self, conns, weights, biases, use_softmax, nodes):
        self.conns = conns              # lijst van tuples: (from_node, to_node)
        self.weights = weights          # dict: (from_node, to_node): weight
        self.biases = biases            # dict: node_id: bias_value
        self.use_softmax = use_softmax  # boolean
        self.nodes = nodes              # lijst van tuples: (node_id, "in"/"out"/"hidden")
        self.input_size = 0
        self.output_size = 0
        for node in nodes:
            if node[1] == "in":
                self.input_size += 1
            if node[1] == "out":
                self.output_size += 1
        self.conn_values = {}          # dict: (from_node, to_node): value      (input * weight) 
        
    def softmax(self, inputs):
        max_val = max(inputs)
        exp_waarden = [math.exp(x - max_val) for x in inputs]
        som_exp = sum(exp_waarden)
        return [geexp / som_exp for geexp in exp_waarden]

    def forward(self, inputs):
        all_nodes = set()
        for integer, layertype in self.nodes:
            all_nodes.add(integer)
        all_nodes.update(self.biases.keys())
        node_values = {node_id: 0.0 for node_id in all_nodes}
        for i in range(self.input_size):
            node_id = i + 1
            if node_id in node_values:
                node_values[node_id] = inputs[i]
        sorted_nodes = sorted(list(all_nodes))
        for _ in range(2):
            next_values = {node_id: 0.0 for node_id in all_nodes}
            for i in range(self.input_size):
                next_values[i + 1] = inputs[i]
            for from_node, to_node in self.conns:
                signal = node_values[from_node] * self.weights[(from_node, to_node)]
                self.conn_values[(from_node, to_node)] = signal
                next_values[to_node] += signal
            for node_id in sorted_nodes:
                if node_id > self.input_size:
                    bias = self.biases.get(node_id, 0.0)
                    total_input = next_values[node_id] + bias
                    try:
                        node_values[node_id] = 1.0 / (1.0 + math.exp(-total_input))
                    except OverflowError:
                        node_values[node_id] = 0.0 if total_input < 0 else 1.0
                else:
                    node_values[node_id] = next_values[node_id]
        outputs = []
        for q in range(len(node_values) - self.input_size):
            out_id = self.input_size + q + 1
            if out_id in node_values:
                outputs.append(node_values[out_id])
        raw_outputs = outputs[:(len(self.biases) - self.input_size)]
        if self.use_softmax:
            return self.softmax(raw_outputs)
        else:
            return raw_outputs