## bracket_reader.py
## Author: Yangfeng Ji
## Load RST parsing brackets
## Date: 08-29-2016
## Time-stamp: <yangfeng 09/16/2016 11:23:15>


class Elem(object):
    def __init__(self, left_idx, right_idx, label, rela):
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.label = label
        self.rela = rela
        self.left_child = None
        self.right_child = None
        self.nucleus = None
        self.nucleus_edu = None
        self.form = None
        self.is_leaf = False


class BracketReader(object):
    def __init__(self):
        self.parse = None


    def _load_brackets(self, fbracket):
        with open(fbracket, 'r') as fin:
            text = fin.read()
            brackets = text.strip().split("\n")
            brackets = [eval(item) for item in brackets]
        return brackets


    def _construct_elem(self, elem, stack):
        right_child = stack.pop()
        left_child = stack.pop()
        elem.right_child = right_child
        elem.left_child = left_child
        if (left_child.label == "Nucleus") and (right_child.label == "Satellite"):
            elem.form = "NS"
            elem.nucleus = (left_child.left_idx, left_child.right_idx)
            elem.nucleus_edu = left_child.nucleus_edu
        elif (left_child.label == "Nucleus") and (right_child.label == "Nucleus"):
            elem.form = "NN"
            elem.nucleus = (left_child.left_idx, right_child.right_idx)
            elem.nucleus_edu = left_child.nucleus_edu
        elif (left_child.label == "Satellite") and (right_child.label == "Nucleus"):
            elem.form = "SN"
            elem.nucleus = (right_child.left_idx, right_child.right_idx)
            elem.nucleus_edu = right_child.nucleus_edu
        else:
            raise ValueError("what is this?")
        return elem, stack


    def read(self, fbracket):
        self.parse = None # Reset
        brackets = self._load_brackets(fbracket)
        stack = []
        for bracket in brackets:
            t, label, rela = bracket
            left_idx, right_idx = int(t[0]), int(t[1])
            elem = Elem(left_idx, right_idx, label, rela)
            if left_idx == right_idx:
                elem.nucleus = (left_idx, right_idx)
                elem.nucleus_edu = left_idx
                elem.form = "NN"
                elem.is_leaf = True
                stack.append(elem)
            else:
                elem, stack = self._construct_elem(elem, stack)
                stack.append(elem)
        # Combine the last two pieces
        if len(stack) != 2:
            print len(stack)
            raise ValueError("unexpected stack status")
        right_idx = stack[-1].right_idx
        left_idx = stack[-2].left_idx
        elem = Elem(left_idx, right_idx, None, None)
        elem, stack = self._construct_elem(elem, stack)
        self.parse = elem


    def convert(self):
        """ Convert to a dependency structure
        """
        queue = [self.parse]
        deps = [('ROOT', self.parse.nucleus_edu, 'root')]
        while queue:
            elem = queue.pop(0)
            if elem.left_child is not None:
                # if it has left child
                queue.append(elem.left_child)
            if elem.right_child is not None:
                # if it has right child
                queue.append(elem.right_child)
            if elem.is_leaf:
                # pass all leaf nodes
                continue
            if (elem.form == "NS") or (elem.form == "NN"):
                head = elem.left_child.nucleus_edu
                modifier = elem.right_child.nucleus_edu
                rela = elem.right_child.rela
            elif elem.form == "SN":
                head = elem.right_child.nucleus_edu
                modifier = elem.left_child.nucleus_edu
                rela = elem.left_child.rela
            deps.append((head, modifier, rela))
        return deps

        
def test():
    fname = "data/parses/Immigration1.0-28458.brackets"
    reader = BracketReader()
    reader.read(fname)
    deplist = reader.convert()
    for dep in deplist:
        print dep


if __name__ == '__main__':
    test()
