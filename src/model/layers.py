#!/user/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import List, Dict
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def _rindex(sequence, obj) -> int:
    """
    Return zero-based index in the sequence of the last item whose value is equal to obj.  Raises a
    ValueError if there is no such item.

    Parameters
    ----------
    sequence : ``Sequence[T]``
    obj : ``T``

    Returns
    -------
    zero-based index associated to the position of the last item equal to obj
    """
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == obj:
            return i

    raise ValueError(f"Unable to find {obj} in sequence {sequence}.")


def _get_combination(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise ValueError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == '*':
            return first_tensor * second_tensor
        elif operation == '/':
            return first_tensor / second_tensor
        elif operation == '+':
            return first_tensor + second_tensor
        elif operation == '-':
            return first_tensor - second_tensor
        else:
            raise ValueError("Invalid operation: " + operation)


def _get_combination_and_multiply(combination: str,
                                  tensors: List[torch.Tensor],
                                  weight: torch.nn.Parameter) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return torch.matmul(tensors[index], weight)
    else:
        if len(combination) != 3:
            raise ValueError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == '*':
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == '/':
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.pow(-1).transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == '+':
            return torch.matmul(first_tensor, weight) + torch.matmul(second_tensor, weight)
        elif operation == '-':
            return torch.matmul(first_tensor, weight) - torch.matmul(second_tensor, weight)
        else:
            raise ValueError("Invalid operation: " + operation)


def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with :func:`combine_tensors`.  This function computes the resultant dimension when
    calling ``combine_tensors(combination, tensors)``, when the tensor dimension is known.  This is
    necessary for knowing the sizes of weight matrices when building models that use
    ``combine_tensors``.

    Parameters
    ----------
    combination : ``str``
        A comma-separated list of combination pieces, like ``"1,2,1*2"``, specified identically to
        ``combination`` in :func:`combine_tensors`.
    tensor_dims : ``List[int]``
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to :func:`combine_tensors`.
    """
    if len(tensor_dims) > 9:
        raise ValueError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')
    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(',')])


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise ValueError("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise ValueError("Tensor dims must match for operation \"{}\"".format(operation))
        return first_tensor_dim


def combine_tensors_and_multiply(combination: str,
                                 tensors: List[torch.Tensor],
                                 weights: torch.nn.Parameter) -> torch.Tensor:
    """
    Like :func:`combine_tensors`, but does a weighted (linear) multiplication while combining.
    This is a separate function from ``combine_tensors`` because we try to avoid instantiating
    large intermediate tensors during the combination, which is possible because we know that we're
    going to be multiplying by a weight vector in the end.

    Parameters
    ----------
    combination : ``str``
        Same as in :func:`combine_tensors`
    tensors : ``List[torch.Tensor]``
        A list of tensors to combine, where the integers in the ``combination`` are (1-indexed)
        positions in this list of tensors.  These tensors are all expected to have either three or
        four dimensions, with the final dimension being an embedding.  If there are four
        dimensions, one of them must have length 1.
    weights : ``torch.nn.Parameter``
        A vector of weights to use for the combinations.  This should have shape (combined_dim,),
        as calculated by :func:`get_combined_dim`.
    """
    if len(tensors) > 9:
        raise ValueError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')
    pieces = combination.split(',')
    tensor_dims = [tensor.size(-1) for tensor in tensors]
    combination_dims = [_get_combination_dim(piece, tensor_dims) for piece in pieces]
    dims_so_far = 0
    to_sum = []
    for piece, combination_dim in zip(pieces, combination_dims):
        weight = weights[dims_so_far:(dims_so_far + combination_dim)]
        dims_so_far += combination_dim
        to_sum.append(_get_combination_and_multiply(piece, tensors, weight))
    result = to_sum[0]
    for result_piece in to_sum[1:]:
        result = result + result_piece
    return result


class DotProductMatrixAttention(nn.Module):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using a dot
    product.
    """

    @staticmethod
    def forward(matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        return matrix_1.bmm(matrix_2.transpose(2, 1))


class CosineMatrixAttention(nn.Module):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using cosine
    similarity.
    """

    @staticmethod
    def forward(matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        a_norm = matrix_1 / (matrix_1.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        b_norm = matrix_2 / (matrix_2.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        return torch.bmm(a_norm, b_norm.transpose(-1, -2))


# Need check something wrong with return output, without activate func
class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
    and ``Y`` is computed as ``X W Y^T + b``.

    Parameters
    ----------
    matrix_1_dim : ``int``
        The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : ``int``
        The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    use_input_biases : ``bool``, optional (default = False)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    label_dim : ``int``, optional (default = 1)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to ``torch.nn.Bilinear``
        for matrices, rather than vectors.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 use_input_biases: bool = False,
                 label_dim: int = 1) -> None:
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if label_dim == 1:
            self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = Parameter(torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim))

        self._bias = Parameter(torch.Tensor(1))
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        if self._use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], -1)
            matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        return final.squeeze(1) + self._bias


class LinearMatrixAttention(nn.Module):
    """
    This ``MatrixAttention`` takes two matrices as input and returns a matrix of attentions
    by performing a dot product between a vector of weights and some
    combination of the two input matrices, followed by an (optional) activation function.  The
    combination used is configurable.

    If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
    ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
    function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
    bias parameter, and ``[;]`` is vector concatenation.

    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.

    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : ``str``, optional (default="x,y")
        Described above.
    """

    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y') -> None:
        super().__init__()
        self._combination = combination
        combined_dim = get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    def forward(self,  # pylint: disable=arguments-differ
                matrix_1: torch.Tensor,
                matrix_2: torch.Tensor) -> torch.Tensor:
        combined_tensors = combine_tensors_and_multiply(self._combination,
                                                             [matrix_1.unsqueeze(2), matrix_2.unsqueeze(1)],
                                                             self._weight_vector)
        return combined_tensors + self._bias


class ElementWiseMatrixAttention(nn.Module):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.

    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """
    def __init__(self) -> None:
        super(ElementWiseMatrixAttention, self).__init__()

    @staticmethod
    def forward(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = torch.einsum('iaj,ibj->ijab', [tensor_1, tensor_2])
        return result


class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
    3D tensor.

    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
    and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
    it to every time step.
    """
    def forward(self, input_tensor):
        # pylint: disable=arguments-differ
        """
        Apply dropout to input tensor.

        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``

        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = F.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor],
                        num_wrapping_dims: int = 0) -> torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
    wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
    is given by ``num_wrapping_dims``.

    If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
    If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
    dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
    if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.

    If the input ``text_field_tensors`` contains the "mask" key, this is returned instead of inferring the mask.

    TODO(joelgrus): can we change this?
    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.ByteTensors  makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    # >>> mask = torch.ones([260]).byte()
    # >>> mask.sum() # equals 260.
    # >>> var_mask = torch.autograd.V(mask)
    # >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    if "mask" in text_field_tensors:
        return text_field_tensors["mask"]

    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))
