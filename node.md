# Why do we need automatic differentiation?
Without automatic differentiation, we need to implement backpropogation (to calcualte derivatives) manually. With automatic differentiation, we only need to implement forward propogation of the model and backpropogation is done by calling backward() function of loss.

# Data structures
## TensorOp:
1. call: create a new tensor and use op.compute data of the tensor.
2. compute: compute array operation.
3. gradient: return gradients of each input.

## Tensor
op, inputs, cached_data, requires_grad, grad  
1. make_from_op: create a new tensor and op.compute()
2. make_const: create a new tensor
3. overloaded mathematic methods: can be invoked with mathematic operators like + - * /.

# Forward
Construct forward computational graph
1. Tensor(): convert input X and y to tensors
2. Construct model and loss function with ops_mathematic:  
ops.call -> Tensor.make_from_op (create a new tensor) -> Tensor.realize_cached_data -> op.compute  
New tensors are created automatically while doing operations. Inputs of each tensor are stored in it.  
The final tensor is loss.

# Backward
1. Find_topo_sort: get an ordered list of nodes (tensors).
2. Use algorithm in the lecture to compute gradients and store them in tensors in the graph.

# Questions
Why is the computational graph just used for one time. New tensors of w1 and w2 are created in simple_ml.py/nn_epoch per batch. So, new computational graph is constructed each batch. Why not to update w1 and w2's data and invoke realize_cached_data function of all nodes? 