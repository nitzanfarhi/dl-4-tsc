#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix,find
import math
from matplotlib import pyplot as plt


def create_grid(row,col):
    n = row*col
    adj_mat = np.zeros((row*col, row*col))
    #print(n)
    for i in range(0,n,2):
        if i//col%2 !=0:
            continue
        adj_mat[i][i +col] = 1
        adj_mat[i+col][i]=1

        adj_mat[i][i + 1] = 1
        adj_mat[i+1][i] = 1

        adj_mat[i+col][i +col+1] = 1
        adj_mat[i+col+1][i+col]=1

        adj_mat[i+1][i +col+1] = 1
        adj_mat[i+col+1][i+1]=1


    grid_graph = csr_matrix(adj_mat)
    return grid_graph

def ul(num,row,col,n):
    cir_col = col // 2
    cir_row = row // 2
    return ((num//cir_col)*2*col+(num%cir_row)*2)
def ur(num,row,col,n):
    return ul(num,row,col,n)+1
def ll(num,row,col,n):
    return ul(num,row,col,n)+col
def lr(num,row,col,n):
    return ur(num,row,col,n)+col

def count_edges(grid_graph):
    print(len(grid_graph.nonzero()[1]))

def assign(grid_graph,src,dst,f1,f2,row,col,n):
    #print("Add",src,dst,f1,f2,f1(src,row,col,n),f2(dst,row,col,n))
    grid_graph[f1(src,row,col,n),f2(dst,row,col,n)]=1
    grid_graph[f2(dst,row,col,n),f1(src,row,col,n)]=1

    #count_edges(grid_graph)
def remove(grid_graph,src,dst,f1,f2,row,col,n):
    #print("Remove",src,dst,f1,f2,f1(src,row,col,n),f2(dst,row,col,n))
    grid_graph[f1(src,row,col,n),f2(dst,row,col,n)]=0
    grid_graph[f2(dst,row,col,n),f1(src,row,col,n)]=0
    #count_edges(grid_graph)
    
    
def create_circuit(row,col,weights):
    n = row*col
    circuit_mat = np.zeros((n//2,n//2))

    cir_row = row // 2
    cir_col = col // 2
    n = cir_row*cir_col
    circuit_mat = np.zeros((cir_row*cir_col, cir_row*cir_col))
    weight_counter = 0
    for i in range(n):
        if i < (cir_row-1)*cir_col:
            circuit_mat[i][i +cir_col] =  weights[weight_counter]
            circuit_mat[i+cir_col][i]= weights[weight_counter]
            weight_counter+=1
        if i % cir_col < cir_col-1:
            circuit_mat[i][i + 1] =  weights[weight_counter]
            circuit_mat[i+1][i] =  weights[weight_counter]
            weight_counter+=1
    return circuit_mat

def create_path(grid_graph,circuit_mat,row,col):
    n = row*col
    cir_row = row // 2
    cir_col = col // 2
    circuit_graph = csr_matrix(circuit_mat)
    edges_list = list([tuple(mrow) for mrow in np.transpose(minimum_spanning_tree(circuit_graph).nonzero())])
    addinf = (row,col,n)
    for src,dst in edges_list:
        if dst-src == 1:
            # RIGHT
            remove(grid_graph,src,src,ur,lr,*addinf)
            remove(grid_graph,dst,dst,ul,ll,*addinf)

            assign(grid_graph,src,dst,ur,ul,*addinf)
            assign(grid_graph,src,dst,lr,ll,*addinf)

        if dst-src == -1:
            # LEFT
            remove(grid_graph,src,src,ll,ul,*addinf)
            remove(grid_graph,dst,dst,lr,ur,*addinf)

            assign(grid_graph,src,dst,ll,lr,*addinf)
            assign(grid_graph,src,dst,ul,ur,*addinf)

        if dst-src == -cir_col:
            # UP
            remove(grid_graph,src,src,ur,ul,*addinf)
            remove(grid_graph,dst,dst,lr,ll,*addinf)

            assign(grid_graph,src,dst,ul,ll,*addinf)
            assign(grid_graph,src,dst,ur,lr,*addinf)

        if dst-src == cir_col:
            # DOWN
            remove(grid_graph,src,src,ll,lr,*addinf)
            remove(grid_graph,dst,dst,ur,ul,*addinf)

            assign(grid_graph,src,dst,ll,ul,*addinf)
            assign(grid_graph,src,dst,lr,ur,*addinf)

    first_node = 0
    semi_result_path = [first_node]
    counter = 1
    while len(semi_result_path)<row*col:
        r,c,_ = find(grid_graph[first_node])
        for next_node in c:
            if next_node not in semi_result_path:
                semi_result_path.append(next_node)
                first_node = next_node
                break
        if counter+1!=len(semi_result_path):
            #print("Path Error",r,c)
            return
        else:
            counter+=1
    result_x = []
    result_y = []
    for value in semi_result_path:
        result_x.append(value%col)
        result_y.append(value//col)
    return result_x,result_y

def create_path_from_shape(row,col,weights):
    grid_graph = create_grid(row,col)
    circuit_mat=create_circuit(row,col, weights)
    return create_path(grid_graph,circuit_mat,row,col)


# In[15]:


    

def moore(cycles):
    AXIOM = 'LFL+F+LFL'

    RULES = {
        'L': '-RF+LFL+FR-',
        'R': '+LF-RFR-FL+',
    }
    cycles = int( math.sqrt(cycles))

    def draw(string,cycles):

        left_dict = {"L":"D","D":"R","R":"U","U":"L"}
        right_dict = {"L":"U","U":"R","R":"D","D":"L"}
        forward_dict = {"L":(-1,0),"D":(0,1),"R":(1,0),"U":(0,-1)}
        result_x = []
        result_y = []
        x_pos = 0
        y_pos = 0
        result_x.append(x_pos)
        result_y.append(y_pos)
        pos = 'R'
        for character in string:
            if character == 'F':
                x_adv,y_adv =forward_dict[pos]
                x_pos+=x_adv
                y_pos+=y_adv
                result_x.append(x_pos)
                result_y.append(y_pos)
            elif character == '+':
                pos = right_dict[pos]
            elif character == '-':
                pos = left_dict[pos]
            else:
                pass  # ignore other characters
        return np.array(result_x),np.array(result_y)+abs(min(result_y))

    def produce(string):
        production = ''

        for character in string:
            if character in RULES:
                production += RULES[character]
            else:
                production += character  # just copy other characters

        return production


    string = AXIOM

    for _ in range(1, cycles):
        string = produce(string)
    return draw(string,cycles)

def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield(x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield(x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        yield from generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2))

        


        
def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)
        
        


def closest_square(n):
    return int(math.pow(2,math.ceil(math.log(n)/math.log(2))))

def run_iterative(alg,row,col):
    length = closest_square(row*col)
    print(length,row*col)
    length-=1
    x = np.zeros(length, dtype=np.int16)
    y = np.zeros(length, dtype=np.int16)
    for i in range(0, length):
        x[i], y[i] = alg(length, i)
    res_curve = (x,y)
    #show_example(x,y)
    return res_curve

def run_overall(alg):
    x,y = alg(n)
    #show_example(x,y)
    return x,y



def readucr(filename):
    data = np.loadtxt(filename, delimiter='\t')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def create_curves(row,col):
    n = row
    for x in [gilbert,hilbert,sweep,scan,zcurve]:
        yield run_iterative(x,row,col)
        
    
    #predefined_curves += [run_overall(x) for x in [moore]]
    #for curve in predefined_curves:
        yield curve

def gilbert_curve(row,col):
    gilbert_x = []
    gilbert_y = []
    for x,y in gilbert2d(col,row):
        gilbert_x.append(x)
        gilbert_y.append(y)
    return gilbert_x,gilbert_y

import sys
import time
print("Started "+str(time.time()))
SCALE = 10
START_OFFSET = 350

def part1by1(n):
    n &= 0x0000ffff
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def unpart1by1(n):
    n &= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n


def interleave2(x, y):
    return part1by1(x) | (part1by1(y) << 1)


def zcurve(_,n):
    return unpart1by1(n), unpart1by1(n >> 1)


def rot(n, x, y, rx, ry):
    """
    rotate/flip a quadrant appropriately
    """
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y

def hilbert(n, d):
    t = d
    x = y = 0
    s = 1
    while (s < n):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t = t // 4
        s *= 2
    return x, y


def one_sweep(row,col,d):
    return d%row, d // col

def one_scan(row,col,d):
    cur_y = d //row
    if cur_y %2 ==0:
        return d%col, cur_y
    else:
        return col - (d%col)-1, cur_y
    
def scan(row,col):
    return zip(*[one_scan(row,col,i) for i in range(row*col)])
    
def sweep(row,col):
    return zip(*[one_sweep(row,col,i) for i in range(row*col)])

def random(row,col):
    m_total_weights = ((col-1)*(row-1)*2)+(col-1)+(row-1)
    return create_path_from_shape(row,col,np.random.random(m_total_weights))



def data_generator(path,totalX,row,col,verbose=False):
    x_pos, y_pos =  path
    images = []
    x_res = []
    y_res = []
    feature_num = 1 if len(totalX.shape) <= 2 else totalX.shape[2]
    imgs = []
    for x_vals in totalX:
        #batch_x = []
        #batch_y = []
        if feature_num > 1:
            final_img = np.zeros((len(np.unique(x_pos))+1,len(np.unique(y_pos))+1,feature_num))
        else:
            final_img =  np.zeros((len(np.unique(x_pos))+1,len(np.unique(y_pos))+1))
        for cur_x, cur_y, value in zip(x_pos, y_pos, x_vals):
                if  math.isnan(value):
                    value = 0
                #print(cur_x,cur_y,value)
                final_img[cur_x][cur_y] = value
                
        final_img[np.isnan(final_img)] = 0
        #plt.imshow(final_img)
        if feature_num > 1:
            X_transform = np.zeros((final_img.shape[0], final_img.shape[1], 3))
            for j in range(final_img.shape[1]):
                pca = PCA(n_components=3)
                f  = pca.fit_transform(final_img[j,:])            
                X_transform[j, :] = f

            final_img = X_transform.reshape(-1,X_transform.shape[0],X_transform.shape[1])
        else:
            final_img = final_img.reshape(final_img.shape[0],final_img.shape[1])

        imgs.append(final_img)
    return np.array(imgs)
            #batch_x.append(final_img)
        #batch_x,batch_y =  np.array(batch_x),np.array(batch_y)    
        #yield batch_x,batch_y 

"""
m_row = 16
m_col = 16
m_cir_row = m_row // 2
m_cir_col = m_col // 2
m_total_weights = ((m_cir_col-1)*(m_cir_row-1)*2)+(m_cir_col-1)+(m_cir_row-1)

create_path_from_shape(m_row,m_col,np.random.random(m_total_weights))
"""

def show_actual(img):
    fig = plt.figure(figsize=(3, 3), facecolor='w')
    plt.imshow(img, cmap='hot', interpolation='nearest')
    ax = fig.add_axes([0.05,0.9,0.9,0.05])
    ax.set_xlim([-0.5, (1 << 1) - .5])
    ax.set_ylim([-(1 << 1) + .5, 0.5])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.box(on=False)
    plt.show()
    
def show_example(x,y):
    fig = plt.figure(figsize=(3, 3), facecolor='w')
    plt.plot(x, y)
    ax = fig.add_axes([0.05,0.9,0.9,0.05])
    ax.set_xlim([-0.5, (1 << 1) - .5])
    ax.set_ylim([-(1 << 1) + .5, 0.5])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.box(on=False)
    plt.show()
    