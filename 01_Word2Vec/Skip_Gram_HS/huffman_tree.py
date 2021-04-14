class HuffmanNode:
    """
    Define the Node class of HuffmanTree
    """

    def __init__(self, word_id, frequency):
        self.word_id = word_id
        self.frequency = frequency
        self.code = []
        self.path = []
        self.left_child = None
        self.right_child = None
        self.father = None


class HuffmanTree:
    def __init__(self, word_frequency_dict):
        self.word_count = len(word_frequency_dict)
        self.wordid_code = dict()
        self.wordid_path = dict()
        self.root = None
        unmerge_node_list = [HuffmanNode(wordid, frequency) for wordid, frequency in word_frequency_dict.items()]
        self.huffman = [HuffmanNode(wordid, frequency) for wordid, frequency in word_frequency_dict.items()]
        print("Building huffman tree...")
        self.build_tree(unmerge_node_list)
        print("Builidng tree finished")

        print("Generating huffman path...")
        self.generate_huffman_code_and_path()
        print("Generating huffman path finished")

    def merge_node(self, node1, node2):
        """
        Merge two leaves to an father node
        :param node1: leaf node1
        :param node2: leaf node2
        :return: father node
        """
        sum_frequency = node1.frequency + node2.frequency
        father_node_id = len(self.huffman)
        father_node = HuffmanNode(father_node_id, sum_frequency)
        if node1.frequency >= node2.frequency:
            father_node.left_child = node1
            father_node.right_child = node2
        else:
            father_node.left_child = node2
            father_node.right_child = node1
        self.huffman.append(father_node)
        return father_node

    def build_tree(self, node_list):
        """
        Choose every two min-value leaf node to merge
        :param node_list: unmerge_list
        :return: tree structure
        """

        while len(node_list) > 1:  # if the deque of leaves is larger than one, at least 2
            i1 = 0  # the minimum frequency node
            i2 = 1  # the second minimum frequency node
            if node_list[i1].frequency > node_list[i2].frequency:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        [i1, i2] = [i2, i1]
            father_node = self.merge_node(node_list[i1], node_list[i2])
            if i2 > i1:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i2 < i1:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError("i1 should not be equal to i2")
            node_list.insert(0, father_node)
        self.root = node_list[0]

    def generate_huffman_code_and_path(self):
        """
        pre-order traversal to generate the code and path of the tree
        :return:
        """
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            while node.left_child or node.right_child:
                code = node.code
                path = node.path
                node.left_child.code = code + [1]
                node.right_child.code = code + [0]
                node.left_child.path = path + [node.word_id]
                node.right_child.path = path + [node.word_id]
                # put the right child tree into stack
                stack.append(node.right_child)
                node = node.left_child
            # Here remains the leaf node need to handle
            word_id = node.word_id
            word_code = node.code
            word_path = node.path
            self.huffman[word_id].code = word_code
            self.huffman[word_id].path = word_path
            self.wordid_code[word_id] = word_code
            self.wordid_path[word_id] = word_path

    def get_all_pos_and_neg_path(self):
        positive = []
        negative = []
        for word_id in range(self.word_count):
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.huffman[word_id].code):
                if code == 1:
                    pos_id.append(self.huffman[word_id].path[i])
                else:
                    neg_id.append(self.huffman[word_id].path[i])
            positive.append(pos_id)
            negative.append(neg_id)
        return positive, negative


if __name__ == "__main__":
    word_frequency = {0: 4, 1: 6, 2: 3, 3: 2, 4: 2}
    print(word_frequency)
    tree = HuffmanTree(word_frequency)
    print(tree.wordid_code)
    print(tree.wordid_path)
    for i in range(len(word_frequency)):
        print(tree.huffman[i].path)
    pos,neg = tree.get_all_pos_and_neg_path()
    print(pos,neg)