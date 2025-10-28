#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:56:00 2025

@author: denniswei, sadhamanus
"""

import torch
import torch._dynamo
import numpy as np
import torch.nn.init as init
import math
from torch import nn
from transformers.models.cofrnets.cofrnet_modules.Customized_Linear_Classes import CustomizedLinearFunction
from transformers.models.cofrnets.cofrnet_modules.Customized_Linear_Classes import CustomizedLinear
from transformers.models.cofrnets.cofrnet_modules.utils_CoFrNet import generate_connections
from transformers.models.cofrnets.cofrnet_modules.utils_CoFrNet import modified_reciprocal_activation, MinMaxClipper

torch._dynamo.config.suppress_errors = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoFrContinuant(torch.autograd.Function):
    """
    PyTorch Function to compute continued fractions and their gradients given partial denominators
    (and optionally partial numerators)
    """

    #Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, a, b=None, epsilon=0.01):#, mask=None, output_direct=None):
        """
        Compute continued fractions given partial denominators (and optionally partial numerators)

        Parameters
        ----------
        ctx : 
            PyTorch context object
        a : (batch_size, seq_len, width, depth) Tensor
            Partial denominators
        b : (batch_size, seq_len, width, depth) Tensor or None
            Partial numerators
        epsilon : float
            Parameter that caps reciprocal function at 1/epsilon in magnitude

        Returns
        -------
        output : (batch_size, seq_len, width) Tensor
            Continued fraction outputs

        """

        #a_copy = a.clone()

        # Dimensions
        batch_size, seq_len, width, depth = a.shape
        

        # Compute continuants K_{d-k} = K[:, :, k] for k = d, d-1, ..., 0
        K = torch.ones(batch_size, seq_len, width, depth + 1, dtype=a.dtype, device=a.device)
        K[:, :, :, depth - 1] = a[:, :, :, depth - 1]
        for k in range(depth - 2, -1, -1):
            if b is None:
                # Numerators = 1 so no need to multiply by them
                K[:, :, :, k] = a[:, :, :, k] * K[:, :, :, k + 1] + K[:, :, :, k + 2]
            else:
                K[:, :, :, k] = a[:, :, :, k] * K[:, :, :, k + 1] + b[:, :, :, k + 1] * K[:, :, :, k + 2]

        # Evaluate function
        K0_reciprocal = modified_reciprocal_activation(K[:, :, :, 0], epsilon)
        output = K[:, :, :, 1] * K0_reciprocal  # / K[:, :, :, 0]
        if b is not None:
            output *= b[:, :, :, 0]

        # Save variables to context
        ctx.save_for_backward(b, K, K0_reciprocal)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients of continued fractions w.r.t. partial denominators (and optionally partial numerators)

        Parameters
        ----------
        ctx : 
            PyTorch context object
        grad_output : (batch_size, seq_len, width) Tensor
            Output gradients

        Returns
        -------
        grad_a : (batch_size, seq_len, width, depth) Tensor
            Gradients w.r.t. partial denominators
        grad_b : (batch_size, seq_len, width, depth) Tensor or None
            Gradients w.r.t. partial numerators
        grad_epsilon : None
            No gradient for hyperparameter epsilon

        """
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        #b, K, K0_reciprocal, mask, output_direct = ctx.saved_tensors
        #grad_a = grad_b = grad_mask = None

        b, K, K0_reciprocal = ctx.saved_tensors
        grad_a = grad_b = None
        

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            # Divide K by K_d = K[:, 0] as everything depends on these ratios
            K_ratio = K * K0_reciprocal.unsqueeze(-1)  # / K[:, :, :, [0]]

            # Gradient w.r.t. a
            grad_a = K_ratio[:, :, :, 1:] ** 2

            if b is None:
                # Numerators = 1 so just need to negate gradients of odd layers
                grad_a[:, :, :, ::2] = grad_a[:, :, :, ::2].neg()
            else:
                # Multiply by cumulative product of b
                p = torch.ones_like(K)
                p[:, :, :, 1:] = torch.cumprod(-b, dim=3)
                grad_a *= p[:, :, :, 1:]

            grad_a *= grad_output.unsqueeze(-1)
        
        if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
            # Gradient w.r.t. b - NEEDS CHECKING
            grad_b = p[:, :, :, :-1] * K[:, :, :, :-1] * K[:, :, :, 1:]
            grad_b *= grad_output.unsqueeze(-1)
        
        #return grad_a, grad_b, None, None, grad_output_direct
        return grad_a, grad_b, None
"""
class DiagonalLinear(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super(DiagonalLinear, self).__init__()
        self.in_features_per_block = in_features
        self.out_features_per_block = out_features // num_blocks
        #self.num_blocks = num_blocks
        #self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        #self.bias = nn.Parameter(torch.Tensor(out_features))
        #self.reset_parameters()
        self.connections = generate_connections(num_blocks, self.in_features_per_block, self.out_features_per_block, 'diagonalized', self.in_features_per_block)
        #self.layers = nn.ModuleList()
        self.layers = torch.tensor(self.connections[0]).to(device)
        #self.layers.append(CustomizedLinear(torch.tensor(self.connections[0])))

        for i in range(1, len(self.connections)):
            self.layers = torch.cat((self.layers, torch.tensor(self.connections[i]).to(device)), dim=0)
            #self.layers.append(CustomizedLinear(torch.tensor(self.connections[i])))
    
    '''
    def reset_parameters(self):
      mask = torch.eye(self.in_features_per_block, dtype=torch.bool)


      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      init.uniform_(self.bias, -bound, bound) 
     
      
      self.weight.requires_grad_(False)
      
      with torch.no_grad():
          for i in range(self.num_blocks):
              start_row = i * self.in_features_per_block
              end_row = (i+1) * self.in_features_per_block
              start_col = 0
              end_col = self.in_features_per_block
          
              self.weight[start_row:end_row, start_col:end_col].masked_fill_(~mask, 0) #Fill non-diagonals with 0s
              #torch.diagonal(self.weight[start_row:end_row, start_col:end_col]).requires_grad_(True) #Enable gradients for diagonals           
    '''      
    def forward(self, x):
        #print(f'x size: {x.size()}, self layers size: {self.layers.size()}')
        return torch.einsum('ijk, lk-> ijl', x, self.layers) #torch.nn.functional.linear(x, self.layers, self.bias)
"""

class DiagonalLinear(nn.Module):
    def __init__(self, input_dim, depth):
        super(DiagonalLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, depth))
        self.bias = nn.Parameter(torch.Tensor(input_dim, depth))
        self.reset_parameters()

    def reset_parameters(self):
      
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      
      
      init.constant_(self.weight, 1e-6)
      init.constant_(self.bias, 1e6)

      '''
      init.uniform_(self.bias, -bound, bound)
      #init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      init.uniform_(self.weight, -bound, bound)
      

      init.normal_(self.bias, 0, bound) #1.5307
      init.normal_(self.weight, 0, bound)
      
          
      init.xavier_uniform_(self.bias) #1.5434
      init.xavier_uniform_(self.weight)
      
      init.xavier_normal_(self.bias)
      init.xavier_normal_(self.weight)
      
      init.kaiming_uniform_(self.bias, a=math.sqrt(5))
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      
      init.kaiming_normal_(self.bias, a=math.sqrt(5))
      init.kaiming_normal_(self.weight, a=math.sqrt(5))
      '''
    def forward(self, x):
        # computes partial denominators a
        # assumes x.shape[-1] == input_dim, for example x.shape = [batch_size, seq_len, input_dim] after transposing
        a = x.unsqueeze(-1) * self.weight + self.bias
        #if torch.isnan(self.bias).any():
         #       raise Exception("Tensor bias has NaNs")
        return a    # a.shape = [batch_size, seq_len, input_dim, depth]



class UpperTriangularLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(UpperTriangularLinear, self).__init__()
        '''
        self.linear = nn.Parameter(torch.Tensor(in_features, out_features)) #nn.Linear(in_features, out_features, bias).to(device)
        self.bias = nn.Parameter(torch.Tensor(in_features)) 
        self.reset_parameters()
        self.mask = torch.triu(torch.ones_like(self.linear)).bool().to(device)
        self.bias_flag = bias

        def mask_grad_hook(grad):
            return grad*self.mask
        
        self.linear.register_hook(mask_grad_hook)
        '''
        connections = generate_connections(1, in_features, out_features, 'upper_triangular')
        self.layers = nn.ModuleList()
        self.layers.append(CustomizedLinear(torch.tensor(connections[0])))
        

    def reset_parameters(self):
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.linear, -bound, bound)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        #print(f'prod size: {self.linear.size()}, bias size: {self.bias.size()}')
        #return x @ (self.linear * self.mask) + self.bias.unsqueeze(0).unsqueeze(0) if self.bias_flag == True else x @ (self.linear * self.mask)
        return self.layers[0](x, False)



class CoFrNetContinuant(nn.Module):
    """
    PyTorch Module for a CoFrNet combining affine transformations and a CoFrContinuant
    """

    def __init__(self, input_dim, output_dim, width=1, depth=1, bias=True, epsilon=0.01, clip_cont_frac = True, generalized=False, variant='fully_connected'):
        """
        Initialize CoFrNetContinuant

        Parameters
        ----------
        input_dim : int
            Number of input dimensions
        output_dim : int
            Number of output dimensions
        width : int
            Number of continued fractions (hidden dimensions)
        depth : int
            Depth of continued fractions
        bias : bool
            Include bias term in linear layers (default True)
        epsilon : float
            Parameter that caps reciprocal function at 1/epsilon in magnitude
        clip_cont_frac : bool
            Clip based on epsilon value or not
        generalized : bool
            Use generalized continued fractions with non-unit numerators (default False)

        Returns
        -------
        CoFrNetContinuant

        """
        super(CoFrNetContinuant, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width 
        self.depth = depth
        self.bias = bias
        self.epsilon = epsilon
        self.clip_cont_frac = clip_cont_frac
        self.generalized = generalized
        #self.mask = torch.triu(torch.ones(input_dim, output_dim)).bool()
        
        if variant == 'fully_connected':
            # Direct affine component from input to output
            self.linear_direct = nn.Linear(input_dim, output_dim, bias)
            
            if self.width and self.depth:
                # Affine transformation for partial denominators
                self.linear_den = nn.Linear(input_dim, self.width * self.depth, bias)
                #init.constant_(self.linear_den.weight, 1e-6)
                #init.constant_(self.linear_den.bias, 1e6)

                if self.generalized:
                    # Affine transformation for partial numerators
                    self.linear_num = nn.Linear(input_dim, self.width * self.depth, bias)

                if clip_cont_frac:
                    # MinMaxClipper for continued fraction outputs
                    self.clipper = MinMaxClipper(self.width)

                # Linear transformation of continued fraction outputs
                self.linear_output = nn.Linear(self.width, output_dim, bias=False)
        elif variant == 'diagonalized_causal': #width is input_dim
            # Direct affine component from input to output

            self.linear_direct = UpperTriangularLinear(input_dim, output_dim, bias)
            
            if self.width and self.depth:
                # Affine transformation for partial denominators
                #self.linear_den = DiagonalLinear(input_dim, self.width * self.depth, self.depth)
                self.linear_den = DiagonalLinear(input_dim, self.depth)

                if self.generalized:
                    # Affine transformation for partial numerators
                    #self.linear_num = DiagonalLinear(input_dim, self.width * self.depth, self.depth)
                    self.linear_num = DiagonalLinear(input_dim, self.depth)

                if clip_cont_frac:
                    # MinMaxClipper for continued fraction outputs
                    self.clipper = MinMaxClipper(self.width)

                # Linear transformation of continued fraction outputs
                self.linear_output = UpperTriangularLinear(self.width, output_dim, bias)




    def forward(self, x, input_dim, variant='fully_connected'):
        """
        Compute CoFrNetContinuant output

        Parameters
        ----------
        x : (batch_size, seq_len, input_dim) Tensor
            Inputs

        Returns
        -------
        output : (batch_size, seq_len, output_dim) Tensor
            Outputs

        """
        #print(f'x unpadded dim: {x.size(2)}, input dim: {input_dim}')
        if (variant == 'diagonalized_causal') and (x.size(2) < input_dim):
            sizediff = input_dim - x.size(2)
            x = torch.nn.functional.pad(x, (0, sizediff), mode='constant', value=0) #Required for sampling to match dimension of weight matrix when attention is replaced
            #print(f'x padded dim: {x.size(2)}')

        # Compute direct affine component of output
        output = self.linear_direct(x)
        output = output.clone()

        if self.width and self.depth:
            # Compute partial denominators
            a = self.linear_den(x)
            a = a.reshape(-1, x.size(dim=1), self.width, self.depth)
            
            if self.generalized:
                # Compute partial numerators
                b = self.linear_num(x)
                b = b.reshape(-1, x.size(dim=1), self.width, self.depth)
                # Compute continued fractions
                cont_frac = CoFrContinuant.apply(a, b, self.epsilon)
            else:
                # Compute continued fraction with unit numerators
                cont_frac = CoFrContinuant.apply(a, None, self.epsilon)
            
            
            if self.clip_cont_frac:
                # Clip continued fraction outputs (evaluation mode) or update minimum and maximum values (training mode)
                cont_frac = self.clipper(cont_frac)
            

            # Add linear combination of continued fractions to output
            if variant == 'fully_connected':
                output += self.linear_output(cont_frac)
            elif variant == 'diagonalized_causal':
                #print(f'cont_frac: {cont_frac.size()}, output: {output.size()}')
                output += self.linear_output(cont_frac)

        return output

    def reset_cont_frac_min_max(self):
        """
        Reset minimum and maximum values of continued fraction outputs

        """
        if self.clip_cont_frac:
            # Reset min/max values for each clipper
            self.clipper.reset_min_max()
        else:
            print("No MinMaxClippers to reset because clip_cont_frac=False")

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, width={self.width}, depth={self.depth}, bias={self.bias}, epsilon={self.epsilon}, clip_cont_frac={self.clip_cont_frac}, generalized={self.generalized}'