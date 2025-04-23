import itertools
import os
from sortedcontainers import SortedSet
from typing import List, Dict, Set, Tuple


class Dfg:
    def __init__(self):
        self.ninputs = 0
        self.ndata = 0
        self.nexs = 0
        self.nsels = 0
        self.fmulti = False
        self.datanames = []
        self.operands = []
        self.nodes = []
        self.outputnames = []
        self.oprs = []
        self.name2node = {}
        self.multioprs = []
        self.operands_ = []
        self.oprtypes = []
        self.dependents = []
        self.unique = {}
        self.priority = []
        self.exs = []
        self.exconds = []

    class Node:
        def __init__(self):
            self.type = 0
            self.vc = []
            self.id = 0
            self.dependent = False
            self.exind = []
        def __init__(self, id=0, type=0, vc=None, exind=None):
            self.id = id
            self.type = type
            self.vc = vc if vc is not None else []
            self.exind = exind if exind is not None else []
            self.dependent = False

    class Opr:
        def __init__(self, s: str, n: int, attr: int):
            self.s = s
            self.n = n
            self.attr = attr

    def get_ninputs(self) -> int:
        return self.ninputs

    def get_ndata(self) -> int:
        # print('lmao')
        return self.ndata

    def get_fmulti(self) -> bool:
        return self.fmulti

    def get_dataname(self, i: int) -> str:
        return self.datanames[i]

    def get_operands(self) -> List[List[Set[int]]]:
        return self.operands

    def get_priority(self) -> List[Tuple[int, int, bool]]:
        return self.priority

    def get_nexs(self) -> int:
        return self.nexs

    def get_nsels(self) -> int:
        return self.nsels

    def get_exs(self) -> List[Tuple[bool, int, int, List[int]]]:
        return self.exs

    def get_exconds(self) -> List[List[Dict[int, bool]]]:
        return self.exconds

    def show_error(self, msg: str, value: str):
        raise ValueError(f"{msg}: {value}")

    def create_opr(self, s: str, n: int, fc: bool = False, fa: bool = False):
        attr = 0
        if fc:
            attr += 1
        if fa:
            attr += 2
        self.oprs.append(self.Opr(s, n, attr))

    def oprtype(self, s: str) -> int:
        for i, opr in enumerate(self.oprs):
            if opr.s == s:
                return i
        return -1

    def typeopr(self, i: int) -> str:
        if i < 0 or i >= len(self.oprs):
            self.show_error("no oprtype", str(i))
        return self.oprs[i].s

    def fcommutative(self, i: int) -> bool:
        if i < 0 or i >= len(self.oprs):
            self.show_error("no oprtype", str(i))
        return self.oprs[i].attr % 2 != 0

    def fassociative(self, i: int) -> bool:
        if i < 0 or i >= len(self.oprs):
            self.show_error("no oprtype", str(i))
        return (self.oprs[i].attr >> 1) % 2 != 0

    def create_input(self, name: str):
        self.datanames.append(name)
        p = self.Node()
        p.type = -1
        p.id = self.ninputs
        self.ninputs += 1
        self.name2node[name] = p
        self.nodes.append(p)

    def create_node(self, vs: list[str], pos: list[int]) -> 'Node':
        # Kiểm tra nếu vị trí vượt quá kích thước của vs
        if pos[0] >= len(vs):
            raise ValueError(f"Incomplete line: {' '.join(vs)}")
        opr_idx = self.oprtype(vs[pos[0]])
        if opr_idx < 0:
            # Trường hợp không phải operator, tìm trong name2node
            if vs[pos[0]] not in self.name2node:
                self.show_error("unspecified data", vs[pos[0]])
            return self.name2node[vs[pos[0]]]

        # Tạo một Node mới
        p = self.Node()
        p.id = -1  # Đặt id mặc định là -1
        p.type = opr_idx  # Loại operator

        # Lặp qua số lần thực hiện của operator này và tạo các con
        for _ in range(self.oprs[p.type].n):
            # Tạo các node con, đồng thời cập nhật pos
            pos[0] +=1
            c = self.create_node(vs, pos)  # Chuyển pos dưới dạng list
            p.vc.append(c)

        # Thêm node vào danh sách nodes
        self.nodes.append(p)
        return p

    def create_node_for_list(self, vs: list[str]):
        # print(vs)
        if vs[0] in self.name2node:
            self.show_error("duplicated data", vs[0])
        pos = [1]  # Dùng danh sách để lưu giá trị của pos
        self.name2node[vs[0]] = self.create_node(vs, pos)

    def create_multiopr(self, vs: list[str], pos: list[int]) -> 'Node':
        # Kiểm tra nếu vị trí vượt quá kích thước của vs
        if pos[0] >= len(vs):
            raise ValueError(f"Incomplete line: {' '.join(vs)}")

        opr_idx = self.oprtype(vs[pos[0]])
        if opr_idx < 0:
            return None

        # Tạo một Node mới
        p = self.Node()
        p.id = 1  # Đặt id mặc định là 1
        p.type = opr_idx  # Loại operator

        # Lặp qua số lần thực hiện của operator này và tạo các con
        for _ in range(self.oprs[p.type].n):
            # Tạo các node con, đồng thời cập nhật pos
            pos[0]+=1
            c = self.create_multiopr(vs, pos)  # Chuyển pos dưới dạng list
            if c:
                p.id = 0  # Cập nhật lại id nếu có node con
            p.vc.append(c)

        # Thêm node vào danh sách nodes
        self.nodes.append(p)
        return p

    def create_multiopr_for_list(self, vs: list[str]):
        pos = [0]  # Dùng danh sách để lưu giá trị của pos
        p = self.create_multiopr(vs, pos)
        self.multioprs.append(p)

    def insert_xbtree_node(self, p):
        if p.type < 0:
            return
        
        if len(p.vc) > 2 and self.fassociative(p.type) and self.fcommutative(p.type):
            q = self.Node(id=-1, type=-3, vc=p.vc, exind=[])
            self.nodes.append(q)
            
            if len(p.vc) == 3:
                r = self.Node(id=-1, type=p.type, vc=[q], exind=[0])
                r.vc.append(q)
                r.exind.append(1)
                self.nodes.append(r)
                
                p.vc = [q]
                p.exind = [2]
                p.vc.append(r)
            
            elif len(p.vc) == 4:
                r = self.Node(id=-1, type=p.type, vc=[q], exind=[0])
                r.vc.append(q)
                r.exind.append(1)
                self.nodes.append(r)
                
                s = self.Node(id=-1, type=-2, vc=[q], exind=[2])
                s.vc.append(r)
                self.nodes.append(s)
                
                t = self.Node(id=-1, type=p.type, vc=[q], exind=[3])
                t.vc.append(s)
                t.exind.append(0)
                self.nodes.append(t)
                
                p.vc = [s]
                p.exind = [1]
                p.vc.append(t)
            
            elif len(p.vc) == 5:
                r = self.Node(id=-1, type=p.type, vc=[q,q], exind=[0,1])
                self.nodes.append(r)
                
                s = self.Node(id=-1, type=-2, vc=[q,r], exind=[2])
                self.nodes.append(s)
                
                t = self.Node(id=-1, type=p.type, vc=[q,s], exind=[3,0])
                self.nodes.append(t)
                
                u = self.Node(id=-1, type=-2, vc=[q,t], exind=[4])
                self.nodes.append(u)
                
                v = self.Node(id=-1, type=p.type, vc=[s,u], exind=[1,0])
                self.nodes.append(v)
                
                p.vc = [u]
                p.exind = [1]
                p.vc.append(v)
            else:
                self.show_error("currently more than 7 input xbtree is not supported")
            
            p = q
        
        for c in p.vc:
            self.insert_xbtree_node(c)

    def insert_xbtree(self):
        for s in self.outputnames:
            p = self.name2node.get(s)
            if not p:
                self.show_error("unspecified output", s)
            self.insert_xbtree_node(p)

    def support_multiopr_rec(self, id, ope, vs, vm):
        if ope is None:
            # print('each')
            for s in vs:
                s.add(id)
                # print(s)
            return True

        if id >= 0 and (self.dependents[id] or self.oprtypes[id] != ope.type):
            return False

        vs_ = []
        vm_ = []

        if id < 0:
            id = -id - 1
            exid = len(self.exs) - 1
            for i in range(len(self.exs)):
                if self.exs[i][1] > id:
                    exid = i - 1
                    break
            
            ex = self.exs[exid]
            ind = id - ex[1]
            nc = len(ex[3])

            for ii in range(nc):
                vs2 = [SortedSet()]
                vm2 = [{}]
                s = ex[2]

                if ex[0]:
                    for ii_ in range(nc):
                        vm2[0][s + ind * nc + ii_] = ii == ii_
                    for i in range(nc):
                        vm2[0][s + i * nc + ii] = i == ind
                else:
                    for ii_ in range(nc):
                        vm2[0][s + ii_] = ii_ == (ii - ind + nc) % nc

                r = self.support_multiopr_rec(ex[3][ii], ope, vs2, vm2)
                if not r:
                    continue

                for i in range(len(vs2)):
                    vs_.append(vs2[i])
                    vm_.append(vm2[i])

        else:
            for v in self.operands_[id]:
                if self.fcommutative(ope.type) and ope.id == 0:
                    for indices in itertools.permutations(list(range(self.oprs[ope.type].n))):
                        vs2 = [SortedSet()]
                        vm2 = [{}]
                        for i in range(self.oprs[ope.type].n):
                            r = self.support_multiopr_rec(v[indices[i]], ope.vc[i], vs2, vm2)
                            if not r:
                                break
                        
                        if r:
                            for i in range(len(vs2)):
                                vs_.append(vs2[i])
                                vm_.append(vm2[i])

                else:
                    vs2 = [SortedSet()]
                    vm2 = [{}]
                    r = False
                    for i in range(self.oprs[ope.type].n):
                        r = self.support_multiopr_rec(v[i], ope.vc[i], vs2, vm)
                        if not r:
                            break
                    
                    if not r:
                        continue
                    
                    for i in range(len(vs2)):
                        vs_.append(vs2[i])
                        vm_.append(vm2[i])

        vsnew = []
        vmnew = []

        for i in range(len(vs)):
            for j in range(len(vs_)):
                m = vm[i].copy()
                f = False
                for e in vm_[j].items():
                    if e[0] in m and m[e[0]] != e[1]:
                        f = True
                        break
                    m[e[0]] = e[1]
                
                if f:
                    continue
                
                s = vs[i].copy()
                # print('a',s)
                s.update(vs_[j])
                # print(s)
                vsnew.append(s)
                vmnew.append(m)

        if not vsnew:
            return False

        vs.clear()
        vm.clear()
        vs.extend(vsnew)
        vm.extend(vmnew)
        return True

    def support_multiopr(self):
        for multiopr in self.multioprs:
            for i in range(self.ndata):
                # print('yes')
                vs = [SortedSet()]
                vm = [{}]
                tmp = self.dependents[i]
                self.dependents[i] = False
                # print(i, multiopr.id)
                r = self.support_multiopr_rec(i, multiopr, vs, vm)
                self.dependents[i] = tmp
                # print(r)
                if r:
                    for j in range(len(vs)):
                        # print(vs[j])
                        self.operands[i].append(vs[j])
                        self.exconds[i].append(vm[j])


    def gen_operands_node(self, p, fname):
        if p.id != -1:
            return

        cids = []
        for i in range(len(p.vc)): 
            c = p.vc[i] 
            self.gen_operands_node(c, fname) 
            if c.type < -1:  
                cids.append(-(c.id + 1 + p.exind[i]))  
            else:
                cids.append(c.id) 

        if p.type < -1: 
            p.id = self.nexs 
            self.exs.append((p.type == -3, self.nexs, self.nsels, cids)) 
            self.nexs += len(cids)
            if p.type == -2:
                self.nsels += len(cids)
            elif p.type == -3: 
                self.nsels += len(cids) * len(cids)
            else:
                self.show_error("unexpected error")
            return

        if self.fcommutative(p.type):  
            cids.sort()

        key = (p.type, tuple(cids)) 
        if key in self.unique:
            p.id = self.unique[key] 
            return

        if not self.fassociative(p.type): 
            sv = {tuple(cids)}
            self.oprtypes.append(p.type) 
            self.dependents.append(p.dependent)  
            self.operands_.append(sv)

            dataname = ""
            if fname:
                dataname += self.typeopr(p.type) + " "  
                for cid in cids:
                    if cid < 0:
                        dataname += f"__ex{-cid-1} "
                    else:
                        dataname += f"{self.datanames[cid]} "
                dataname = dataname.strip()

            self.datanames.append(dataname)
            self.unique[key] = self.ndata
            self.ndata += 1
            p.id = self.unique[key]  # Sử dụng p.id thay vì p['id']
            return

        if not self.fcommutative(p.type): 
            for i in range(2, len(cids) + 1):
                for j in range(len(cids) - i + 1):
                    sub = cids[j:j + i]
                    keysub = (p.type, tuple(sub)) 
                    if keysub in self.unique:
                        continue
                    sv = SortedSet()
                    for k in range(1, i):
                        a = sub[:k]
                        b = sub[k:]
                        v = []
                        if len(a) == 1:
                            v.append(a[0])
                        else:
                            keya = (p.type, tuple(a))  
                            v.append(self.unique[keya])

                        if len(b) == 1:
                            v.append(b[0])
                        else:
                            keyb = (p.type, tuple(b)) 
                            v.append(self.unique[keyb])

                        v.sort()
                        sv.add(tuple(v))

                    self.oprtypes.append(p.type)  
                    self.dependents.append(p.dependent)  
                    self.operands_.append(sv)

                    dataname = ""
                    if fname:
                        dataname += self.typeopr(p.type) + " "  
                        for id in sub:
                            if id < 0:
                                dataname += f"__ex{-id-1} "
                            else:
                                dataname += f"{self.datanames[id]} "
                        dataname = dataname.strip()

                    self.datanames.append(dataname)
                    self.unique[keysub] = self.ndata
                    self.ndata += 1

            p.id = self.unique[key] 
            return

        for i in range(2, len(cids) + 1):
            for indices in self.foreach_comb(len(cids), i):
                sub = [cids[index] for index in indices]
                sub.sort()
                keysub = (p.type, tuple(sub))  
                if keysub in self.unique:
                    continue
                sv = SortedSet()
                for j in range(1, (1 << (i - 1))):
                    v = []
                    a = []
                    b = []
                    j_ = j
                    for k in range(i):
                        if j_ % 2:
                            a.append(sub[k])
                        else:
                            b.append(sub[k])
                        j_ >>= 1

                    if len(a) == 1:
                        v.append(a[0])
                    else:
                        a.sort()
                        keya = (p.type, tuple(a))  
                        v.append(self.unique[keya])

                    if len(b) == 1:
                        v.append(b[0])
                    else:
                        b.sort()
                        keyb = (p.type, tuple(b))  
                        v.append(self.unique[keyb])

                    v.sort()
                    sv.add(tuple(v))

                self.oprtypes.append(p.type) 
                self.dependents.append(p.dependent) 
                self.operands_.append(sv)

                dataname = ""
                if fname:
                    dataname += self.typeopr(p.type) + " " 
                    for id in sub:
                        if id < 0:
                            dataname += f"__ex{-id-1} "
                        else:
                            dataname += f"{self.datanames[id]} "
                    dataname = dataname.strip()

                self.datanames.append(dataname)
                self.unique[keysub] = self.ndata
                self.ndata += 1

        p.id = self.unique[key]
        # print(p.id, self.operands_[p.id])

    def foreach_comb(self, n, k):
        # Hàm tạo các kết hợp của k chỉ số từ 0 đến n-1
        from itertools import combinations
        return combinations(range(n), k)

    def gen_operands(self, fmultiopr, fname):
        self.ndata = self.ninputs
        self.oprtypes.clear()
        self.oprtypes = [None] * self.ndata
        self.dependents.clear()
        self.dependents = [None] * self.ndata
        self.operands_ = []
        self.operands_ = [[] for _ in range(self.ndata)]  # Equivalent to `resize(ndata)` in C++
        self.unique.clear()

        # Process each output name
        for s in self.outputnames:
            p = self.name2node.get(s)
            if not p:
                self.show_error("unspecified output", s)
            self.gen_operands_node(p, fname)

        self.fmulti = 0
        self.operands.clear()
        self.operands = [[] for _ in range(self.ndata)]  # Equivalent to `resize(ndata)` in C++
        self.exconds.clear()
        self.exconds = [[] for _ in range(self.ndata)]  # Empty dict for each element

        # If `fmultiopr` is true, support multi-operation
        if fmultiopr:
            self.support_multiopr()

        # Process the operands and exconds
        for i in range(self.ndata):
            for v in self.operands_[i]:
                s = SortedSet(v)  
                # print(i, v, s)
                m = {} 
                self.operands[i].append(s)
                self.exconds[i].append(m)
                # self.operands[i].reverse()
                # self.exconds[i].reverse()

            s = SortedSet()
            for j in range(len(self.operands[i])):
                # sorted_dict = sorted(self.exconds[i][j].items())
                # s.add((self.operands[i][j]), (self.exconds[i][j].items())) 
                s.add((frozenset(self.operands[i][j]),frozenset(self.exconds[i][j].items())))
                # print(frozenset(sorted_dict))

            self.operands[i].clear()
            self.exconds[i].clear()
            s2 = []
            for e in s:
                s2.append((sorted(e[0]), sorted(e[1])))
            s2 = sorted(s2, key=lambda x: x[0])
            for e in s2:
                # print(e)
                self.operands[i].append(SortedSet(e[0]))
                self.exconds[i].append(dict(sorted(dict(e[1]).items())))

            if not self.fmulti and len(self.operands[i]) > 1:
                self.fmulti = 1


    def print_node(self, p: 'Node', depth: int, exind: int = 0):
        # In ra các tab theo độ sâu
        print("\t" * depth, end="")

        # Kiểm tra loại node
        if p.type == -1:
            print(p.id)
        elif p.type < 0:
            print(f"{p.type}({p}, {exind})")
        else:
            print(self.typeopr(p.type))

        # Nếu exind không bằng 0, kết thúc
        if exind:
            return

        # Đệ quy in các node con
        for i, c in enumerate(p.vc):
            if i < len(p.exind):
                self.print_node(c, depth + 1, p.exind[i])
            else:
                self.print_node(c, depth + 1)

    def print(self):
        # In ra id và tên
        print("id to name:")
        for i, name in enumerate(self.datanames):
            print(f"\t{i} : {name}")

        # In ra cây biểu thức
        print("expression tree:")
        for s in self.outputnames:
            print(f"\t{s} :")
            p = self.name2node.get(s)
            if not p:
                self.show_error("unspecified output", s)
            self.print_node(p, 1)

    def print_operands(self):
        for i in range(self.ninputs, self.ndata):
            a = self.operands[i]
            print(f"{i} :")
            for j in range(len(a)):
                b = a[j]
                print("\t", end="")
                for c in b:
                    print(f"{c:3}", end=", ")
                d = self.exconds[i][j]
                if d:
                    print("\t(", end="")
                    for key, value in d.items():
                        if value:
                            print(f"{key:3}", end=", ")
                    print(")", end="")
                print()

    def read(self, filename):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(script_dir, filename)
            with open(filename, 'r') as f:
                r = True
                while True:
                    if r:
                        line = f.readline().strip()
                    if not line:
                        break
                    
                    r = True
                    
                    vs = line.split()
                    if not vs:
                        continue
                        
                    if vs[0] == ".i":
                        for name in vs[1:]:
                            self.create_input(name)
                    elif vs[0] == ".o":
                        self.outputnames.extend(vs[1:])
                    elif vs[0] == ".f":
                        while True:
                            line = f.readline().strip()
                            if not line:
                                break
                            vs = line.split()
                            if vs[0][0] == ".":
                                r = False
                                break
                            if len(vs) < 2:
                                self.show_error("Incomplete line", line)
                            n = int(vs[1])  # Chuyển đổi số lượng toán hạng
                            if n <= 0:
                                self.show_error("Negative number of operands", vs[1])
                            fc = "c" in vs[2:]
                            fa = "a" in vs[2:]
                            self.create_opr(vs[0], n, fc, fa)
                    elif vs[0] == ".m":
                        while True:
                            line = f.readline().strip()
                            if not line:
                                break
                            vs = line.split()
                            if vs[0][0] == ".":
                                r = False
                                break
                            self.create_multiopr_for_list(vs)
                    elif vs[0] == ".n":
                        while True:
                            line = f.readline().strip()
                            if not line:
                                break
                            vs = line.split()
                            if vs[0][0] == ".":
                                r = False
                                break
                            self.create_node_for_list(vs)
                    elif vs[0] == ".p":
                        while True:
                            line = f.readline().strip()
                            if not line:
                                break
                            vs = line.split()
                            if vs[0][0] == ".":
                                r = False
                                break
                            if len(vs) < 3:
                                self.show_error("Incomplete line", line)
                            p = self.name2node.get(vs[0])
                            if not p:
                                self.show_error("Unspecified data", vs[0])
                            p.dependent = True
                            for i in range(1, len(vs), 2):
                                q = self.name2node.get(vs[i + 1])
                                if not q:
                                    self.show_error("Unspecified data", vs[i + 1])
                                q.dependent = True
                                if vs[i] == ">":
                                    self.priority.append((p, q, False))
                                elif vs[i] == ">=":
                                    self.priority.append((p, q, True))
                                elif vs[i] == "<":
                                    self.priority.append((q, p, False))
                                elif vs[i] == "<=":
                                    self.priority.append((q, p, True))
                                elif vs[i] == "=":
                                    self.priority.append((p, q, True))
                                    self.priority.append((q, p, True))
                                else:
                                    self.show_error("Unknown comparison operator", vs[i])
                                p = q
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    def compress_node(self, p):
        # Nếu node có type < 0, không làm gì cả
        if p.type < 0:
            return

        # Nếu node có tính kết hợp (associative)
        if self.fassociative(p.type):
            vcn = []
            i = 0
            while i < len(p.vc):
                c = p.vc[i]
                if not c.dependent and p.type == c.type:
                    p.vc[i + 1:i + 1] = c.vc
                else:
                    vcn.append(c)
                i += 1
            p.vc = vcn

        # Tiếp tục đệ quy cho các node con
        for c in p.vc:
            self.compress_node(c)

    def compress(self):
        # Duyệt qua danh sách outputnames và gọi compress_node cho mỗi node tương ứng
        for s in self.outputnames:
            p = self.name2node.get(s)
            if not p:
                self.show_error("unspecified output", s)
            self.compress_node(p)

    def update_datanames(self, exmap):
        for i in range(self.ndata):
            if not self.datanames[i]:
                continue  # Nếu dataname trống thì bỏ qua
            
            # Tách chuỗi thành các từ (dựa trên dấu cách ' ')
            vs = []
            s = self.datanames[i]
            parts = s.split(' ')  # Tách chuỗi thành các phần
            for part in parts:
                if len(part) > 4 and part[:4] == "__ex":
                    vs.append(part)
            
            # Thay thế các phần "__ex" bằng giá trị từ exmap
            for s in vs:
                exid = int(s[4:])  # Lấy id sau "__ex"
                datanames_to_replace = self.datanames[exmap[exid]]
                self.datanames[i] = self.datanames[i].replace(s, datanames_to_replace)

    def input_id(self, name: str) -> int:
        # Tìm node từ tên trong từ điển name2node
        p = self.name2node.get(name)

        # Nếu không tìm thấy node, báo lỗi
        if not p:
            self.show_error("unspecified input", name)

        # Nếu node không phải là input hợp lệ, báo lỗi
        if p.id < 0 or p.id >= self.ninputs:
            self.show_error("non-input data", name)

        # Trả về id của node
        return p.id

    def show_error(self, message: str, value: str):
        # Hàm hiển thị lỗi và ném ngoại lệ
        print(f"Error: {message}: {value}")
        raise ValueError(f"{message}: {value}")

    def output_ids(self):
        ids = set()  # Sử dụng set trong Python để lưu trữ các ID duy nhất
        for s in self.outputnames:
            p = self.name2node.get(s)
            if not p:
                self.show_error("unspecified output", s)
            ids.add(p.id)  # Thêm ID của node vào set
        return ids

if __name__ == "__main__":
    d = Dfg()
    d.read("f.txt")
    d.compress()
    d.insert_xbtree()
    d.gen_operands(True,True)
    # print(d.fcommutative,d.fassociative)
    d.print_operands()
    # d.print()
    # d.get_ndata
    # print(d.get_exs())