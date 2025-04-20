import os
from sortedcontainers import SortedSet
class Graph:
    def __init__(self):
        self.nnodes = 0
        self.name2id = {}
        self.nodes = {}
        self.edges = {}
        self.incoms = {}
    
    def get_nnodes(self):
        return self.nnodes
    
    def get_nodes(self):
        # return list(self.nodes["mem"].union(self.nodes["pe"]))
        return self.nodes
    
    def get_pes(self):
        return list(self.nodes["pe"])
    
    def get_mems(self):
        return list(self.nodes["mem"])
    
    def get_edges(self):
        return self.edges['com']
    
    def get_id(self, name):
        return self.name2id.get(name, -1)
    
    def get_id2name(self):
        return {v: k for k, v in self.name2id.items()}
    
    def get_type(self, id_):
        for type_, nodes in self.nodes.items():
            if id_ in nodes:
                return type_
        return ""
    
    def get_types(self):
        return list(self.nodes.keys())
    
    def create_node(self, type_, name):
        if name in self.name2id:
            raise ValueError(f"node duplication: {name}")
        self.nodes.setdefault(type_, set()).add(self.nnodes)
        self.name2id[name] = self.nnodes
        self.nnodes += 1
    
    def read(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, filename)
        with open(filename, 'r') as f:
            r = True
            while True:
                if r:
                    line = f.readline()
                    if not line:
                        break
                    vs = line.strip().split()
                    if not vs:
                        continue
                # print(vs)
                r = True
                if not vs[0].startswith('.'):
                    raise ValueError(f"unexpected line: {line}")
                
                type_ = vs[0][1:]
                if len(vs) != 1:
                    for name in vs[1:]:
                        self.create_node(type_, name)
                    continue
                
                v = []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    
                    vs = line.strip().split()
                    if not vs:
                        continue
                    
                    if vs[0].startswith('.'):
                        r = False
                        break
                    
                    i = 0
                    senders = set()
                    while i < len(vs) and vs[i] != '->':
                        if vs[i] not in self.name2id:
                            raise ValueError(f"unspecified node: {vs[i]}")
                        senders.add(self.name2id[vs[i]])
                        i += 1
                    
                    if i == len(vs):
                        raise ValueError(f"incomplete line: {line}")
                    
                    i += 1
                    recipients = set()
                    while i < len(vs) and vs[i] != ':':
                        if vs[i] not in self.name2id:
                            raise ValueError(f"unspecified node: {vs[i]}")
                        recipients.add(self.name2id[vs[i]])
                        i += 1
                    
                    band = 0
                    if i < len(vs) and vs[i] == ':':
                        i += 1
                        if i == len(vs):
                            raise ValueError(f"incomplete line: {line}")
                        try:
                            band = int(vs[i])
                        except ValueError:
                            raise ValueError(f"non-integer weight: {vs[i]}")
                    
                    v.append((senders, recipients, band))
                    
                
                self.edges.setdefault(type_, []).extend(v)
                
        for i in range(len(self.edges["com"])):
            # print(self.edges["com"][i])
            for recipients in self.edges["com"][i][1]:
                if recipients not in self.incoms:
                    self.incoms[recipients] = SortedSet()
                self.incoms[recipients].add(i)
        # for i in self.incoms.keys():
        #     self.incoms[i] = list(self.incoms[i])
        
    def print(self):
        print("id to name :")
        for name, id_ in self.name2id.items():
            print(f"\t{id_} : {name}")
        
        for type_, nodes in self.nodes.items():
            print(f"{type_} :")
            print("\t", end="")
            print(", ".join(map(str, nodes)))
            print()
        
        for type_, edges in self.edges.items():
            print(f"{type_} :")
            for senders, recipients, band in edges:
                print("\t", end="")
                print(" ".join(map(str, senders)), end=" ")
                print("-> ", end="")
                print(" ".join(map(str, recipients)), end="")
                if band > 0:
                    print(f" : {band}", end="")
                print()
        
                
if __name__ == '__main__':

    
    g = Graph()
    g.create_node("mem", "_extmem")
    g.read("e.txt")
    g.print()
    print(g.get_edges())