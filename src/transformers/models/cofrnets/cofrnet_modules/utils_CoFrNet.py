'''
author: @ishapuri101, @sadhamanus
'''

import numpy as np # linear algebra
import torch # import main library
import torch.nn as nn

def generate_connections(num_total_layers: int, input_size: int, output_size: int, which_variant: str, num_nodes = float('Inf')): 
    '''
    Args:
    num_total_layers: depth of the ladders 
    input_size: number of input features
    output_size: number of output features 
    which_variant: choose one of CoFrNet Variants to work with:  fully_connected, diagonalized, ladder_of_ladders, or diag_ladder_of_ladder_combined 

    Returns:
    a 3D Matrix of 1s and 0s, where a 1 signifies that a connection exists between two nodes, and a 0 signifies that it doesn't. 
    '''


    def fully_connected_constant(): 
        numLayers_notInclOutput = num_total_layers - 1
        genConns = []

        for i in range(0, numLayers_notInclOutput):
            genConns.append(np.ones([input_size, num_nodes]).tolist())

        genConns.append(np.ones([num_nodes, output_size]).tolist())
        return genConns

    def diagonalized(): 
        #numLayers DOES include output layer

        numLayers_notInclOutput = num_total_layers #- 1
        genConns = []

        for i in range(0, numLayers_notInclOutput):
            genConns.append(np.eye(input_size).tolist())

        #genConns.append(np.triu(np.ones([input_size, output_size]),0).tolist())
        return genConns

    def diagonalized_inc_dim_ladders():
        #numLayers DOES include output layer

        numLayers_notInclOutput = num_total_layers - 1
        genConns = []

        for i in range(0, numLayers_notInclOutput):
            genConns.append(np.eye(input_size).tolist())

        genConns.append(np.triu(np.ones([input_size, output_size]),0).tolist())
        #genConns.append(np.triu(np.ones([output_size, output_size]),0).tolist())
        
        genConns2 = []
        for i in range(0, numLayers_notInclOutput):
            toAppend = np.triu(np.ones([input_size, input_size]))
            toAppend[0, 0] = 0
            genConns2.append(toAppend.tolist())#i added this tolist() on april 20th

        genConns2.append(np.triu(np.ones([input_size, output_size]),0).tolist())
        
        return genConns, genConns2


    def ladder_of_ladders():
        getConns = []
        numLayers_notIncOutput = num_total_layers - 1 #numLayers_notIncOutput = input_size

        for i in range(0, numLayers_notIncOutput):
            toAppend = np.ones([input_size, numLayers_notIncOutput])
            toAppend[:, 0:i] = 0 #toAppend[:, input_size-i:input_size] = 0
            getConns.append(toAppend.tolist())#i added this tolist() on april 20th

        getConns.append(np.ones([numLayers_notIncOutput, output_size]).tolist())

        return getConns

    def diagonalized_ladder_of_ladders_combinedd():
        #numLayers DOES include output layer

        #numLayers = numLadders in this case

        getConns = []
        numLayers_notIncOutput = min(num_nodes, num_total_layers - 1)
        #print(f'Max number of full ladders: {numLayers_notIncOutput}')

        for i in range(0, numLayers_notIncOutput):
            ladderOfLadders = np.ones([input_size, numLayers_notIncOutput])
            ladderOfLadders[:, 0:i] = 0 
            toAppend = np.append(np.eye(input_size), ladderOfLadders, axis = 1)
            getConns.append(toAppend.tolist())
        #print(len(getConns[-1][0]))
        getConns.append(np.ones([len(getConns[-1][0]), output_size]).tolist())
        #getConns.append(np.ones([numLayers_notIncOutput, output_size]).tolist())

        return getConns

    def diag_full_connected():
        #numLayers DOES include output layer

        #numLayers = numLadders in this case

        getConns = []
        numLayers_notIncOutput = num_total_layers - 1
        #print(f'Max number of full ladders: {numLayers_notIncOutput}')

        for i in range(0, numLayers_notIncOutput):
            ladderOfLadders = np.ones([input_size, num_nodes])
            toAppend = np.append(np.eye(input_size), ladderOfLadders, axis = 1)
            getConns.append(toAppend.tolist())
        #print(len(getConns[-1][0]))
        getConns.append(np.ones([len(getConns[-1][0]), output_size]).tolist())
        #getConns.append(np.ones([numLayers_notIncOutput, output_size]).tolist())

        return getConns

    def upper_triangular():
        #numLayers DOES include output layer

        genConns = []

        genConns.append(np.triu(np.ones([input_size, output_size]),0).tolist())
        
        return genConns
    
    if which_variant == "fully_connected":
        return fully_connected_constant()
    elif which_variant == "diagonalized":
        return diagonalized()
    elif which_variant == "ladder_of_ladders":
        return ladder_of_ladders()
    elif which_variant == "diag_ladder_of_ladder_combined":
        return diagonalized_ladder_of_ladders_combinedd()
    elif which_variant == "diag_full_connected":
        return diag_full_connected()
    elif which_variant == "diag_inc_dim_ladders":
        return diagonalized_inc_dim_ladders()
    elif which_variant == "upper_triangular":
        return upper_triangular()
    else:
        raise Exception("You must choose one of the following four choices for which_variant: fully_connected, diagonalized, ladder_of_ladders, diag_ladder_of_ladder_combined, diag_full_connected, diag_inc_dim_ladders or upper_triangular")
                

class MinMaxClipper(nn.Module):
    """
    Clips values at evaluation time to minimum and maximum values seen during training
    """
    def __init__(self, width):
        """
        Initialize MinMaxClipper

        Parameters
        ----------
        width : int
            Number of input (and output) dimensions

        Returns
        -------
        MinMaxClipper

        """
        super(MinMaxClipper, self).__init__()
        self.width = width
        # Initialize buffers for minimum and maximum values
        self.register_buffer("min", torch.full((self.width,), torch.inf))
        self.register_buffer("max", torch.full((self.width,), -torch.inf))
    
    def forward(self, x):
        """
        Clip values (evaluation mode) or update min/max values (training mode)

        Parameters
        ----------
        x : (batch_size, seq_len, width) Tensor
            Inputs

        Returns
        -------
        x : (batch_size, seq_len, width) Tensor
            Clipped inputs (evaluation mode) or unmodified inputs (training mode)

        """
        if self.training:
            with torch.no_grad():
                # Update minimum and maximum values
                self.min = torch.minimum(x.min(dim=0).values.min(dim=0).values, self.min)
                self.max = torch.maximum(x.max(dim=0).values.min(dim=0).values, self.max)
        else:
            # Clip values
            ind_lt_min = x < self.min
            x[ind_lt_min] = self.min.expand_as(x)[ind_lt_min]
            ind_gt_max = x > self.max
            x[ind_gt_max] = self.max.expand_as(x)[ind_gt_max]

        return x
    
    def reset_min_max(self):
        """
        Reset minimum and maximum values

        """
        with torch.no_grad():
            self.min = torch.full_like(self.min, torch.inf)
            self.max = torch.full_like(self.max, -torch.inf)


def modified_reciprocal_activation(Wx, epsilon):
    '''
    Activation function that uses capped 1/x described in paper. Takes in Wx, returns modified activation function
    of Wx
    '''

    denom = torch.where(torch.abs(Wx) < epsilon, torch.sign(Wx)*epsilon, Wx)
    denom = torch.where(denom == 0, epsilon, denom)

    return torch.reciprocal(denom)



def process_data(data_filename, first_column_csv, last_column_csv):
    '''
    Args:
    data_filename: filename of data source
    first_column_csv: index (starting from 0) of first column to include in dataset
    last_column_csv: index (starting from 0) of last column to include in dataset. Use -1 if you want to include all of the columns. 

    Returns:
    tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test
    '''
    
    import pandas as pd
    df=pd.read_csv('datasets/' + data_filename, sep=',',header=0)   
    if last_column_csv != -1: 
        last_column_csv = last_column_csv + 1
    X = df.iloc[:, first_column_csv : last_column_csv].values 
    y = df.iloc[:,-1].values.T

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    from sklearn.preprocessing import StandardScaler
    sc = MinMaxScaler(feature_range=(0,1))
    X = sc.fit_transform(X)
    X.argmax()

    seeds = [1, 10, 100, 555, 9897]
    seed = seeds[2]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=seed)

    #CONVERTING TO TENSOR
    tensor_x_train = torch.Tensor(X_train)
    tensor_x_val = torch.Tensor(X_val)
    tensor_x_test = torch.Tensor(X_test)

    tensor_y_val = torch.Tensor(y_val).long()
    tensor_y_train = torch.Tensor(y_train).long()
    tensor_y_test = torch.Tensor(y_test).long()

    return tensor_x_train, tensor_y_train, tensor_x_val, tensor_y_val, tensor_x_test, y_test

