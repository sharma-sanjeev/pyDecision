###############################################################################

# Required Libraries

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

# Function: Fuzzy DEMATEL
# When multiple respondents are involved, the average values of the left, middle, and right evaluations are computed, forming a new matrix that needs to be provided.
def fuzzy_dematel_method(dataset, size_x = 10, size_y = 10):  
    X_a = np.zeros((len(dataset), len(dataset)))
    X_b = np.zeros((len(dataset), len(dataset)))
    X_c = np.zeros((len(dataset), len(dataset)))
    m_a = np.ones ((len(dataset), len(dataset))) # min
    m_c = np.zeros((len(dataset), len(dataset))) # max
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset)):
            a, b, c  = dataset[i][j]
            X_a[i,j] = a
            X_b[i,j] = b
            X_c[i,j] = c
            if (a < m_a[i,j]):
                m_a[i,j] = a
            if (c > m_c[i,j]):
                m_c[i,j] = c
    # Initial direct relation fuzzy matrix
    max_value = max(np.max(np.sum(X_a, axis=1)),np.max(np.sum(X_a, axis=0)),np.max(np.sum(X_b, axis=1)),np.max(np.sum(X_b, axis=0)),np.max(np.sum(X_c, axis=1)),np.max(np.sum(X_c, axis=0)))
    X_a = X_a / max_value
    X_b = X_b / max_value
    X_c = X_c / max_value

    # The fuzzy total-relation matrix
    X_a_1 = np.linalg.inv(np.identity(X_a.shape[0]) - X_a)
    X_b_1 = np.linalg.inv(np.identity(X_b.shape[0]) - X_b)
    X_c_1 = np.linalg.inv(np.identity(X_c.shape[0]) - X_c)

    T1 = np.matmul(X_a, X_a_1)
    T2 = np.matmul(X_b, X_b_1)
    T3 = np.matmul(X_c, X_c_1)

    # The normalized fuzzy direct-relation matrix

    # CFCS (Converting Fuzzy data into Crisp Scores) method
    m_a = np.min(T1)
    m_c = np.max(T3)
    delta = m_c - m_a

    # Normalization
    xl_ijk = (T1 - m_a)/ delta
    xm_ijk = (T2 - m_a) / delta
    xr_ijk = (T3 - m_a) / delta

    # Compute left (ls) and right (rs) normalized value
    xls_ijk = xm_ijk / (1 + xm_ijk - xl_ijk)
    xrs_ijk = xr_ijk / (1 + xr_ijk - xm_ijk)

    # The crisp total-relation matrix
    # Compute the total normalized crisp value
    x_ijk = ((xls_ijk * (1 - xls_ijk) + xrs_ijk * xrs_ijk)) / (1 - xls_ijk + xrs_ijk)

    # Compute crisp values
    Z = m_a + x_ijk * delta

    # Crisp values
    D = np.sum(Z, axis=1)
    R = np.sum(Z, axis=0)
    D_plus_R  = D + R
    D_minus_R = D - R 
    weights   = (D_plus_R - D_minus_R)/(np.sum(D_plus_R + D_minus_R))
    print('QUADRANT I has the Most Important Criteria (Prominence: High, Relation: High)') 
    print('QUADRANT II has Important Criteira that can be Improved by Other Criteria (Prominence: Low, Relation: High)') 
    print('QUADRANT III has Criteria that are not Important (Prominence: Low, Relation: Low)')
    print('QUADRANT IV has Important Criteria that cannot be Improved by Other Criteria (Prominence: High, Relation: Low)')
    print('')
    plt.figure(figsize = [size_x, size_y])
    plt.style.use('ggplot')
    for i in range(0, len(dataset)):
        if (D_minus_R[i] >= 0 and D_plus_R[i] >= np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 1.0, 0.7),)) 
            print('g'+str(i+1)+': Quadrant I')
        elif (D_minus_R[i] >= 0 and D_plus_R[i] < np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 1.0, 0.7),))
            print('g'+str(i+1)+': Quadrant II')
        elif (D_minus_R[i] < 0 and D_plus_R[i] < np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.7, 0.7),)) 
            print('g'+str(i+1)+': Quadrant III')
        else:
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 0.7, 1.0),)) 
            print('g'+str(i+1)+': Quadrant IV')
    axes = plt.gca()
    xmin = np.amin(D_plus_R)
    if (xmin > 0):
        xmin = 0
    xmax = np.amax(D_plus_R)
    if (xmax < 0):
        xmax = 0
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(D_minus_R)
    if (ymin > 0):
        ymin = 0
    ymax = np.amax(D_minus_R)
    if (ymax < 0):
        ymax = 0
    axes.set_ylim([ymin-1, ymax+1]) 
    plt.axvline(x = np.mean(D_plus_R), linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.axhline(y = 0, linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.xlabel('Prominence (D + R)')
    plt.ylabel('Relation (D - R)')
    plt.show()
    return D_plus_R, D_minus_R, weights

###############################################################################
