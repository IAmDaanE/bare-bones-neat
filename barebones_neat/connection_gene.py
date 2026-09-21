class ConnectionGene:
    def __init__(self, from_node_id, to_node_id, weight, innovation_num, enabled=True):
        self.from_node_id = from_node_id  # Integer ID van de start-node
        self.to_node_id = to_node_id      # Integer ID van de eind-node
        self.weight = weight              # Float gewicht van de connectie
        self.innovation_num = innovation_num  # Uniek historisch ID
        self.enabled = enabled            # Boolean of de connectie actief is

    def clone(self):
        return ConnectionGene(self.from_node_id, self.to_node_id, self.weight, self.innovation_num, self.enabled)