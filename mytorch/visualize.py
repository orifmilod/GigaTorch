from graphviz import Digraph

def draw_graph(node):
    '''Draws a node's dependency graph with graphviz.
       Note: This looks like a depth-first, pre-order traversal, but it's a DAG rather than a tree.
             e.g. one node can be used in multiple downstream nodes.'''
    def _draw_node(node):
        '''Draws / adds a single node to the graph.'''
        # Don't add duplicate nodes to the graph.
        # e.g. if we reach a node twice from its two downstream nodes, only add it once
        if f'\t{id(node)}' in dot.body: return

        # Add the node with the appropriate text
        if node._op is None:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>value = {node.data:.4f}</TD></TR>
                <TR><TD>grad = {node.grad:.4f}</TD></TR>
                <TR><TD>label = {node.label}</TD></TR>
                <TR><TD BGCOLOR="#c9c9c9"><FONT FACE="Courier" POINT-SIZE="12">input</FONT></TD></TR>
            </TABLE>>'''
        else:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>value = {node.data:.4f}</TD></TR>
                <TR><TD>grad = {node.grad:.4f}</TD></TR>
                <TR><TD>label = {node.label}</TD></TR>
                <TR><TD BGCOLOR="#c2ebff"><FONT COLOR="#004261" FACE="Courier" POINT-SIZE="12">{node._op}</FONT></TD></TR>
            </TABLE>>'''
        dot.node(str(id(node)), node_text)

    def _draw_edge(parent, node):
        '''Draws / adds a single directed edge to the graph (parent -> node).'''
        # Don't add duplicate edges to the graph.
        # e.g. if we reach a node twice from its two downstream nodes, only add edges to its parents once
        if f'\t{id(parent)} -> {id(node)}' in dot.body: return

        # Add the edge
        dot.edge(str(id(parent)), str(id(node)))

    def _draw_parents(node):
        '''Traverses recursively, drawing the parent at the child's step (in order to draw the edge).'''
        for parent in node._parents:
            _draw_node(parent)
            _draw_edge(parent, node)
            _draw_parents(parent)

    dot = Digraph(graph_attr={'rankdir': 'BT'}, node_attr={'shape': 'plaintext'})
    _draw_node(node)     # Draw the root / output      
    _draw_parents(node)  # Draw the rest of the graph

    dot.render(filename='output')
    return dot

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._parents:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = f"data={n.data} | grad={n.grad} | label={n.label}", shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.render(filename='output')
    return dot

# def draw_nn(mlp):
    # for layer_number in range(mlp.layers):
        # layer = mlp.layers[layer_number]
        # for neuron_number in range(layer.neurons):
            # neuron = layer[neuron_number]
            # create vertex

            # create edges between vertex and other nodes