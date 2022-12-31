import numpy as np
def reduced_row_echelon(A, b=None):
    '''
    For any m by n matrix, A, the loop will iterate n times through the n columns.
    Each iteration of the loop will
    1) find the values in the column that have all 0s to the left of it (ie which values are pivots)
    2) if no pivots are found, skip to the next column, if multiple pivots are found, pick the first one
    3) subtract multiples of the pivot row from all the other rows to produce 0s above and below the pivot
    The loop terminates when no more pivots can be found
    '''
    if b is not None:
        if A.shape[0] != len(b):
            return ArithmeticError('The number of rows of A must match the number of elements in b')
    
    R = A.astype(float)
    for pivot_column in range(R.shape[1]):
        nonzeros                   = ~np.isclose(R[:,:pivot_column+1], 0)
        nonzeros_in_prior_columns  = np.any(nonzeros[:,:pivot_column], axis = 1)
        nonzeros_in_current_column = nonzeros[:,pivot_column]
        
        pivot_rows = ~nonzeros_in_prior_columns & nonzeros_in_current_column
        if not np.any(pivot_rows):
            continue
        pivot_row = np.argmax(pivot_rows) #if there are multiple pivot rows, pick the first one
        
        pivot                  = R[pivot_row, pivot_column]
        multipliers            = R[:,pivot_column]/pivot
        multipliers[pivot_row] = 0 #subtract a multiple of 0 from the pivot row (ie leave it alone)
        scaled_pivot_rows      = np.outer(multipliers, R[pivot_row,:])
        R                      = R - scaled_pivot_rows
        R[pivot_row,:]         = R[pivot_row,:]/pivot
        
        if b is not None:
            b = b - multipliers * b[pivot_row]
            b[pivot_row] = b[pivot_row]/pivot
            
    if b is None:
        return R
    else:
        return R, b