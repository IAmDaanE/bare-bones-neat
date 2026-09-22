import math

class CompiledGenome:
    def __init__(self, conns, weights, biases, use_softmax):
        self.conns = conns          # lijst van tuples: (from_node, to_node)
        self.weights = weights      # dict: (from_node, to_node): weight
        self.biases = biases        # dict: node_id: bias_value
        self.use_softmax = use_softmax      # boolean
        
    def softmax(self, inputs):
        max_val = max(lijst)
        exp_waarden = [math.exp(x - max_val) for x in lijst]
        som_exp = sum(exp_waarden)
        return [geexp / som_exp for geexp in exp_waarden]

    def forward(self, inputs):
        input_size = len(inputs)
        output_nodes = list(range(input_size + 1, input_size + len(self.biases) - input_size + 1))        
        all_nodes = set()
        for from_node, to_node in self.conns:
            all_nodes.add(from_node)
            all_nodes.add(to_node)            
        all_nodes.update(self.biases.keys())
        node_values = {node_id: 0.0 for node_id in all_nodes}
        for i in range(input_size):
            node_id = i + 1
            if node_id in node_values:
                node_values[node_id] = inputs[i]
        sorted_nodes = sorted(list(all_nodes))
        for _ in range(2):
            next_values = {node_id: 0.0 for node_id in all_nodes}
            for i in range(input_size):
                next_values[i + 1] = inputs[i]
            for from_node, to_node in self.conns:
                signal = node_values[from_node] * self.weights[(from_node, to_node)]
                next_values[to_node] += signal
            for node_id in sorted_nodes:
                if node_id > input_size:
                    bias = self.biases.get(node_id, 0.0)
                    total_input = next_values[node_id] + bias
                    try:
                        node_values[node_id] = 1.0 / (1.0 + math.exp(-total_input))
                    except OverflowError:
                        node_values[node_id] = 0.0 if total_input < 0 else 1.0
                else:
                    node_values[node_id] = next_values[node_id]
        outputs = []
        for q in range(len(node_values) - input_size):
            out_id = input_size + q + 1
            if out_id in node_values:
                outputs.append(node_values[out_id])
        raw_outputs = outputs[:(len(self.biases) - input_size)]
        if self.use_softmax:
            return softmax(raw_outputs)
        else:
            return raw_outputs
    
    def visualize(self, x, y, width, height):
        pass