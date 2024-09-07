# Automatic differentiation

## Principle
### Why do we need automatic differentiation?
Without automatic differentiation, we need to implement backpropogation (to calcualte derivatives) manually. With automatic differentiation, we only need to implement forward propogation of the model and backpropogation is done by calling backward() function of loss.

### Data structures
#### TensorOp:
1. call: create a new tensor and use op.compute data of the tensor.
2. compute: compute array operation.
3. gradient: return gradients of each input.

#### Tensor
op, inputs, cached_data, requires_grad, grad  
1. make_from_op: create a new tensor and op.compute()
2. make_const: create a new tensor
3. overloaded mathematic methods: can be invoked with mathematic operators like + - * /.

### Forward
Construct forward computational graph
1. Tensor(): convert input X and y to tensors
2. Construct model and loss function with ops_mathematic:  
ops.call -> Tensor.make_from_op (create a new tensor) -> Tensor.realize_cached_data -> op.compute  
New tensors are created automatically while doing operations. Inputs of each tensor are stored in it.  
The final tensor is loss.

### Backward
1. loss.backward()
2. compute_gradient_of_variables()
3. Find_topo_sort: get an ordered list of nodes (tensors).
4. Use algorithm in the lecture to compute gradients and store them in tensors in the graph.

### Questions
Why is the computational graph just used for one time? Because, for each batch, new data `x` is used. A new computational graph is created based on the `x` and parameters.

## Questions
### Question 1: Implementing forward computation
For each opetation, implement forward() function, which is easy.

### Question 2: Implementing backward computation
For each operation, implement gradient() function. Base on out_grad, compute gradients of each input. It is to calculate partial derivative of each input. 按正常方式求偏导就行，但是有的时候需要调整一下顺序来保证正确的shape，这是一种实践性的简单的求偏导方法，不是线性代数的正规方法。

### Question 3: Topological sort
Convert computational graph to a list of nodes using post order traverse. According to this order to compute gradients of each node.

### Question 4: Implementing reverse mode differentiation
计算每个节点的gradient。因为一个节点可能向下传导多个节点，所以它的gradient是多个节点对它的偏导的和。根据它的gradient，再去计算inputs的各个偏导。

### Question 5-6
使用automatic differentiation 来训练两层neural network.