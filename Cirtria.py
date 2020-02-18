'''
This script performs the code implementation  of "The Ring of Algebraic Functions on
Persistence Bar Codes", we applying topological data analysis for the solution of the problem:How to differentiate triangles from circles?
'''

__author__ = "Yovani Torres Favier"
__copyright__ = "Copyright (C) 2020 CURLYHUB"
__license__ = "GPL v3"


import os
import sys
import csv
import math
import datetime
import random
import numpy as np
import persim
import ripser
import cechmate as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
from persim import plot_diagrams
from persim.visuals import plot_diagrams
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from joblib import Parallel,parallel_backend as pllb, delayed
from persistence_graphical_tools_Bertrand import *
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from skimage import measure
from sklearn import preprocessing
from scipy.spatial import ConvexHull

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

FILEROOT = r"D:\Yovani\PhD\Master\Lectures\SciProb\Delivery\LongIV\Img_13"
SPACE_SAMPLING_POINTS = 1000


# =====================================================================
def dstriangle(n=100,phi=0,noise=None,):
    """
    Sample `n` data points on a triangle.
    Parameters
    -----------
    n : int
        Number of data points in shape.
    phi : float
        tilt angle.
    """
    # One edge have one addicional point 
    n0 = (n//3) + 1
    n1 = (n//3)
    
    # vertices coordinates
    ax = math.cos(phi)
    ay = math.sin(phi)
    cx = math.cos(phi+2.0*math.pi/3)
    cy = math.sin(phi+2.0*math.pi/3)
    bx = math.cos(phi+4.0*math.pi/3)
    by = math.sin(phi+4.0*math.pi/3)

    # General coordinates of the triangle (perimeter)
    T1x = np.arange(start=ax,stop=bx,step=(bx-ax)/n0)
    T1y = np.arange(start=ay,stop=by,step=(by-ay)/n0)
    T2x = np.arange(start=bx,stop=cx,step=(cx-bx)/n1)
    T2y = np.arange(start=by,stop=cy,step=(cy-by)/n1)
    T3x = np.arange(start=cx,stop=ax,step=(ax-cx)/n1)
    T3y = np.arange(start=cy,stop=ay,step=(ay-cy)/n1)
    Tx = np.concatenate((T1x, T2x,T3x))
    Ty = np.concatenate((T1y, T2y,T3y))
    data = np.vstack((Tx, Ty)).T

    if noise: 
        data += noise * np.random.randn(*data.shape)


    return data.astype(float)

# =====================================================================
def dsphere(n=100, d=2, r=1, noise=None, ambient=None,inrregular=True):
    """
    Sample `n` data points on a d-sphere.
    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    if inrregular:
        data = np.random.randn(n, d+1)
        # Normalize points to the sphere
        data = r * data / np.sqrt(np.sum(data**2, 1)[:, None]) 
        if noise: 
            data += noise * np.random.randn(*data.shape)
        if ambient:
            assert ambient > d, "Must embed in higher dimensions"
            data = embed(data, ambient)
    else:
        data = []
        # Normalize points to the sphere
        angle = np.linspace(0,2*np.pi,n, dtype=np.float64)
        for a in angle:
            data.append([r*math.cos(a),r*math.sin(a)])
        data = np.array(data)
        if noise: 
            data += noise * np.random.randn(*data.shape)
        if ambient:
            assert ambient > d, "Must embed in higher dimensions"
            data = embed(data, ambient)

    return data.astype(float)
# =====================================================================

def formationdataset(end_noise=0.1, angles=[],inrregular=True):
    idx = random.randint(0,5)
    data_noisy_circle = dsphere(d=1, n=100, noise=end_noise,inrregular=inrregular)
    data_noisy_triangle = dstriangle(phi=angles[idx], n=100, noise=end_noise)
    return [data_noisy_circle,data_noisy_triangle]

# =====================================================================

def paralleldataset(m=100,start_noise=0,end_noise=0.1,inrregular=True):
    angles = [0.0,math.pi/7,math.pi/6,math.pi/5,math.pi/4,math.pi/3]    
    noise = np.linspace(start_noise,end_noise,m, dtype=np.float64)
    out  = Parallel(n_jobs=2)(delayed(formationdataset)(end_noise=eps,angles=angles,inrregular=inrregular) for eps in noise)
    return out

# =====================================================================

def pre_features_Cech(data=[]):
    cech_ = cm.Cech(verbose=False) #Go up to 2D homology
    cech_.build(data[0])
    dgm_circle = cech_.diagrams() 
    cech_.build(data[1])
    dgm_triangle = cech_.diagrams()
    return [dgm_circle,dgm_triangle]

# =====================================================================

def pre_features_Rips(data=[]):
    Rip = ripser.Rips(maxdim=1,verbose=False) #Go up to 2D homology
    dgm_circle = Rip.fit_transform(data[0])
    dgm_triangle =  Rip.fit_transform(data[1])
    return [dgm_circle,dgm_triangle]

#===================================================================

# is_left(): tests if a point is Left|On|Right of an infinite line.

#   Input: three points P0, P1, and P2
#   Return: >0 for P2 left of the line through P0 and P1
#           =0 for P2 on the line
#           <0 for P2 right of the line
#   See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons"

def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

#===================================================================

# =====================================================================
# wn_PnPoly(): winding number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: wn = the winding number (=0 only if P is outside V[])

def wn_PnPoly(P, V):
    wn = 0   # the winding number counter

    # repeat the first vertex at end
    V = tuple(V[:]) + (V[0],)


    # loop through all edges of the polygon
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] <= P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
    return wn

# =====================================================================

def check_if_P_isclosed(points):
    Isclosed = True
    for p in points:
        wn = wn_PnPoly(p,points)
        if math.fabs(wn) > 1:
            Isclosed = False
            break
    return Isclosed
# =====================================================================

def parallelfeatures(data_noisy,_doRip=True):
    if(_doRip == True):
        with pllb('threading', n_jobs=2):
            out  = Parallel()(delayed(pre_features_Rips)(data=eps) for eps in data_noisy)
    else:
        out  = Parallel(n_jobs=3)(delayed(pre_features_Cech)(data=eps) for eps in data_noisy)
    return out

# =====================================================================


def index_at_infinitum(arr):
    index = np.where(arr == np.inf)
    return index
# =====================================================================

def clean_infinitum(arr,index):
    arr = np.delete(arr, index[0])
    return arr
# =====================================================================

def topo_features(data):
    v1 = np.array(data[0])
    v2 = np.array(data[1])
    v3 = np.concatenate((v1, v2), axis=0)
    x = np.array(v3[:,0])
    y = np.array(v3[:,1])
    idx = index_at_infinitum(y)
    x=clean_infinitum(x,idx)
    y=clean_infinitum(y,idx)
    ymax = np.amax(y)
    f1= float(np.dot(x, (x - y).T))
    f2= float(np.dot((ymax-y), (x - y).T))
    f3= float(np.dot(x**2, (x - y).T**4))
    f4= float(np.dot((ymax-y)**2, (x - y).T**4))
    out = [f1,f2,f3,f4]
    return np.array(out)
# =====================================================================
def formation_datasets(data):
    X=[]
    Y=[]
    C =[]
    T =[]
    for d in data:
        v1 = topo_features(d[0])
        v2 = topo_features(d[1])
        X.append(v1)
        C.append(v1)
        Y.append(1)
        X.append(v2)
        T.append(v2)
        Y.append(0)
    return np.array(X),np.array(Y),np.array(C),np.array(T)
# =====================================================================
def transformtobarcode(data):
    diag =[]
    dim =0
    for d in data:
        for pair in d:
            diag.append([dim,pair])
        dim= dim + 1
    return diag
# =====================================================================
def make_meshgrid(x, y, z, w, h=.3):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    z: data to base z-axis meshgrid on
    w: data to base w-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy,zz,ww  : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    z_min, z_max = z.min() - 1, z.max() + 1
    w_min, w_max = w.min() - 1, w.max() + 1

    T0 = np.arange(x_min, x_max, h)
    T1 = np.arange(y_min, y_max, h)
    T2 = np.arange(z_min, z_max, h)
    T3 = np.arange(w_min, w_max, h)

    l = len(T0)*len(T1)*len(T2)*len(T3)

    xx, yy,zz,ww = np.meshgrid(T0,T1,T2,T3)
    return xx, yy,zz,ww,x_min, x_max, y_min, y_max ,z_min, z_max,w_min, w_max 
# =====================================================================
def plot_contours( clf,xx, yy,zz,ww,X_Circle,X_Triangle,X_MIN,X_MAX,Y_MIN, Y_MAX,Z_MIN, Z_MAX,type=2,eps=0.1,**params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    """
    plt.close('all')
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel(), ww.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a figure with axes for 3D plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the different input points using 3D scatter plotting
    if type ==0:
        Z = Z[:, 0, :,:]
        b1 = ax.scatter(X_Circle[:, 1], X_Circle[:, 2], X_Circle[:, 3], c='red')
        b2 = ax.scatter(X_Triangle[:, 1], X_Triangle[:, 2], X_Triangle[:, 3], c='green')
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_zlabel("W")
    elif type ==1:
        Z = Z[:, :, 0,:]
        b1 = ax.scatter(X_Circle[:, 0], X_Circle[:, 2], X_Circle[:, 3], c='red')
        b2 = ax.scatter(X_Triangle[:, 0], X_Triangle[:, 2], X_Triangle[:, 3], c='green')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("W")
    elif type ==2:
        Z = Z[:, :, :,0]
        b1 = ax.scatter(X_Circle[:, 0], X_Circle[:, 1], X_Circle[:, 3], c='red')
        b2 = ax.scatter(X_Triangle[:, 0], X_Triangle[:, 1], X_Triangle[:, 3], c='green')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("W")
    elif type ==3:
        Z = Z[:, :, :,0]
        b1 = ax.scatter(X_Circle[:, 0], X_Circle[:, 1], X_Circle[:, 2], c='red')
        b2 = ax.scatter(X_Triangle[:, 0], X_Triangle[:, 1], X_Triangle[:, 2], c='green')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # Plot the separating hyperplane by recreating the isosurface for the distance
    # == 0 level in the distance grid computed through the decision function of the
    # SVM. This is done using the marching cubes algorithm implementation from
    # scikit-image.
    try:
        verts, faces = measure.marching_cubes_classic(Z)
        # Scale and transform to actual size of the interesting volume
        verts = verts * \
            [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / SPACE_SAMPLING_POINTS
        verts = verts + [X_MIN+1, Y_MIN+1, Z_MIN+1]
        # and create a mesh to display
        alpha=0.5
        fc = "orange"
        mesh = Poly3DCollection(verts[faces],alpha = alpha, linewidths=1)
        mesh.set_edgecolor('w')
        mesh.set_facecolor((0, 1, 0,alpha))
        ax.add_collection3d(mesh)


        # Some presentation tweaks
        #ax.set_xlim((-5, 5))
        #ax.set_ylim((-5, 5))
        #ax.set_zlim((-5, 5))
        filename = "image_eps_poly_n_"+ str(eps)+str(type)+".png"
        name_Image = os.path.join(FILEROOT,filename)
        ax.legend([mpatches.Patch(color='orange', alpha=0.3), b1, b2],
                  ["Projected hyperplane", "class of circles",
                   "class of  triangles"],
                  loc="lower left",
                  prop=mpl.font_manager.FontProperties(size=11))
        plt.savefig(name_Image)
    except BaseException as e:
        print('Failed to do something: ' + str(e))
    plt.clf()

# =====================================================================
def plot_perturbed_shape(data,m,type,filename, eps):
    p =0
    if m > 0:
        p = random.randint(0,m-1)
    name_Image = os.path.join(FILEROOT,filename)
    title = "Perturbed triangle with $\epsilon$ =" +str(eps)
    gd = 'g--'
    color='green'
    if type == 0:
        title = "Perturbed circle with $\epsilon$ =" +str(eps)
        gd = '.'
        color='black'
    data_noisy = data[p][type]
    plt.plot( data_noisy[:,0], data_noisy[:,1],gd,color=color)
    plt.xlabel(title)
    plt.axis('equal')
    plt.savefig(name_Image)
    plt.clf()
# =====================================================================
def Without_Condition(eps,doRip=True):
    a = datetime.datetime.now()
    for n in range(0,eps.shape[0]-1):
        data_noisy = paralleldataset(100,start_noise =eps[n], end_noise=eps[n+1],inrregular=False)
        plot_perturbed_shape(data_noisy,-1,0,str(eps[n])+"pertubed_circl_.png",eps[n])
        plot_perturbed_shape(data_noisy,-1,1,str(eps[n])+"pertubed_trian_.png",eps[n])
        data_pre_feature = parallelfeatures(data_noisy,_doRip=doRip)
        cir_bar = transformtobarcode(data_pre_feature[0][0])
        tri_bar = transformtobarcode(data_pre_feature[0][1])
        plot_persistence_barcodeModify(cir_bar, alpha=0.6, height=0.5, title='Persistence barcode for circle $\epsilon$ ='+str(eps[n]),filename=str(eps[n])+"cir_bar.png")
        plot_persistence_barcodeModify(tri_bar, alpha=0.6, height=0.5, title='Persistence barcode for triangle $\epsilon$ ='+str(eps[n]),filename=str(eps[n])+"tri_bar.png")
        X,Y,Ci,Tr= formation_datasets(data_pre_feature)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,random_state=42)

        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 1.0  # SVM regularization parameter
        models = (svm.SVC(kernel='linear', C=C),
                  svm.LinearSVC(C=C, max_iter=50000),
                  svm.SVC(kernel='rbf', gamma=0.8, C=C),
                  svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
        models = (clf.fit(X_train, y_train) for clf in models)

        prediction = []
        q=0
        svm_obj = None
        for clf in models:
            pd = clf.predict(X_test)
            prediction.append(pd)
            if q==0:
                svm_obj =clf 
            q=q+1

        # title for the plots
        titles = ('SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel')
        k=0 
        filename = "eps_poly_1"+ str(eps[n])+".txt"
        data = os.path.join(FILEROOT,filename)
        f= open(data,"w+")
        for y_pred in prediction:
            txt = titles[k] + " with accuracy: "+ str(metrics.accuracy_score(y_test, y_pred)*100)  + " precision: "+ str(metrics.precision_score(y_test, y_pred)*100) + "recall: "  + str(metrics.recall_score(y_test, y_pred)*100) + "\r\n"
            f.write(txt)
            k=k+1
            if k==len(titles):
                break
        f.close()
        #X0, X1,X2,X3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        #xx, yy,zz,ww,x_min, x_max, y_min, y_max ,z_min, z_max,w_min, w_max   = make_meshgrid(X0, X1,X2,X3)
        #try:
        #    for k in range(0,2):
        #        if k==0:
        #            plot_contours(svm_obj,xx, yy,zz,ww,Ci, Tr, y_min, y_max ,z_min, z_max,w_min, w_max, type=k,eps=eps[n])
        #        if k==1:
        #            plot_contours(svm_obj,xx, yy,zz,ww,Ci, Tr,x_min, x_max, z_min, z_max,w_min, w_max, type=k,eps=eps[n])
        #        if k==2:
        #            plot_contours(svm_obj,xx, yy,zz,ww,Ci, Tr,x_min, x_max, y_min, y_max ,w_min, w_max, type=k,eps=eps[n])
        #        if k==3:
        #            plot_contours(svm_obj,xx, yy,zz,ww,Ci, Tr,x_min, x_max, y_min, y_max ,z_min, z_max, type=k,eps=eps[n])
        #except ValueError:
            #print("Oops!  That was no valid!!!.  Try again..." )
            #plt.clf()
        b = datetime.datetime.now()
        c = b - a
        print(c.seconds)
# =====================================================================
def With_Condition(eps_begin=0,eps_end=0.1,steps=100, quantity = 100,verify=False):
    a = datetime.datetime.now()
    eps = np.linspace(eps_begin,eps_end,steps, dtype=np.float64)
    beh = []
    pre_data = []
    dataset = []
    w=0
    min_max_scaler = preprocessing.MinMaxScaler()
    for n in range(0,eps.shape[0]-1):
        #  original noisy data 
        data_noisy = paralleldataset(quantity,start_noise =eps[n], end_noise=eps[n+1],inrregular = False)# all with rips
        if verify:
            is_closed = check_if_P_isclosed(data_noisy[0][0])
            if(is_closed):
                # pre-dataset
                data_pre_feature = parallelfeatures(data_noisy)
                pre_data.append(data_pre_feature)
                # formation of four multisymmetric polynomial features
                X,Y,Ci,Tr= formation_datasets(data_pre_feature)
                #scale our data                
                #X = min_max_scaler.fit_transform(X)
                #Ci = min_max_scaler.fit_transform(Ci)
                #Tr = min_max_scaler.fit_transform(Tr)
                dataset.append([X,Y,Ci,Tr])
                # train/Test Split randomly
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,random_state=42)
                # we create an instance of SVM and fit out data. We scale our
                # data since we want to plot the support vectors
                C = 1.0  # SVM regularization parameter
                models = (svm.SVC(kernel='linear', C=C),
                          svm.LinearSVC(C=C, max_iter=50000),
                          svm.SVC(kernel='rbf', gamma=0.8, C=C),
                          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
                models = (clf.fit(X_train, y_train) for clf in models)
                titles = ('SVC with linear kernel',
                      'LinearSVC (linear kernel)',
                      'SVC with RBF kernel',
                      'SVC with polynomial (degree 3) kernel')
                k=0 
                filename = "eps_poly_1_without"+ str(eps[n])+".txt"
                data = os.path.join(FILEROOT,filename)                        
                accuracy = []
                q=0
                svm_obj = []
                f= open(data,"w+")
                for mod in models:
                    pred = mod.predict(X_test)
                    score = metrics.accuracy_score(y_test, pred)
                    accuracy.append(score)
                    txt = titles[k] + " with accuracy: "+ str(metrics.accuracy_score(y_test, pred)*100) + " precision: "+ str(metrics.precision_score(y_test, pred)*100) + "recall: "  + str(metrics.recall_score(y_test, pred)*100) +  "\r\n"
                    f.write(txt)
                    k=k+1
                    if k==len(titles):
                        break
                f.close()
                maxi_pred = max(accuracy)
                maxi_pred_index= [i for i, j in enumerate(accuracy) if j == maxi_pred]        
                if(len(maxi_pred_index)==4):
                    aux = maxi_pred_index[3]
                    maxi_pred_index =[]
                    maxi_pred_index.append(aux)        
                beh.append([maxi_pred,maxi_pred_index,eps[n],w])
                w = w+1
        else:
            # pre-dataset
            data_pre_feature = parallelfeatures(data_noisy)
            # formation of four multisymmetric polynomial features
            X,Y,Ci,Tr= formation_datasets(data_pre_feature)
            #scale our data 
            #X = min_max_scaler.fit_transform(X)
            #Ci = min_max_scaler.fit_transform(Ci)
            #Tr = min_max_scaler.fit_transform(Tr)
            dataset.append([X,Y,Ci,Tr])
            # train/Test Split randomly
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,random_state=42)
            # we create an instance of SVM and fit out data. We scale our
            # data since we want to plot the support vectors
            C = 1.0  # SVM regularization parameter
            models = (svm.SVC(kernel='linear', C=C),
                        svm.LinearSVC(C=C, max_iter=50000),
                        svm.SVC(kernel='rbf', gamma=0.8, C=C),
                        svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
            models = (clf.fit(X_train, y_train) for clf in models)
            titles = ('SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel')
            k=0 
            filename = "eps_poly_1_without"+ str(eps[n])+".txt"
            data = os.path.join(FILEROOT,filename)                        
            accuracy = []
            q=0
            svm_obj = []
            f= open(data,"w+")
            for mod in models:
                pred = mod.predict(X_test)
                score = metrics.accuracy_score(y_test, pred)
                accuracy.append(score)
                txt = titles[k] + " with accuracy: "+ str(metrics.accuracy_score(y_test, pred)*100) + " precision: "+ str(metrics.precision_score(y_test, pred)*100) + "recall: "  + str(metrics.recall_score(y_test, pred)*100) +  "\r\n"
                f.write(txt)
                k=k+1
                if k==len(titles):
                    break
            f.close()
            maxi_pred = max(accuracy)
            maxi_pred_index= [i for i, j in enumerate(accuracy) if j == maxi_pred]        
            if(len(maxi_pred_index)==4):
                aux = maxi_pred_index[3]
                maxi_pred_index =[]
                maxi_pred_index.append(aux)        
            beh.append([maxi_pred,maxi_pred_index,eps[n],n])

    if len(beh) >0:   
        beh = np.array(beh)  
        b_max = np.amax(np.array([row[0] for row in beh]),axis=0)
        b_min = np.amin(np.array([row[0] for row in beh]),axis=0)
        # set graphical result for the best and the worst result
        b_n = np.where(beh[:,0]==b_max)
        w_n = np.where(beh[:,0]==b_min)
        index = [b_n[0][0],w_n[0][0]]
        if b_max == b_min:
            index = [b_n[0][0],w_n[0][w_n[0].shape[0]-1]]
        _not_best =False;
        desc ="_best_"
        for idx in index:
            if _not_best:
                desc ="_worst_"
            _not_best = True
            dat = beh[idx] 
            BX = dataset[dat[3]]
            if verify:
                cir_bar = transformtobarcode(pre_data[dat[3]][0][0])
                tri_bar = transformtobarcode(pre_data[dat[3]][0][1])
                plot_persistence_barcodeModify(cir_bar, alpha=0.6, height=0.5, title='Persistence barcode for circle $\epsilon$ ='+str(dat[2]),filename=str(dat[2])+ desc + "cir_bar.png")
                plot_persistence_barcodeModify(tri_bar, alpha=0.6, height=0.5, title='Persistence barcode for triangle $\epsilon$ ='+str(dat[2]),filename=str(dat[2])+ desc +"tri_bar.png")
            X_train, X_test, y_train, y_test = train_test_split(BX[0], BX[1], test_size=0.33,random_state=42)
            C = 1.0  # SVM regularization parameter
            models = (svm.SVC(kernel='linear', C=C),
                        svm.LinearSVC(C=C, max_iter=50000),
                        svm.SVC(kernel='rbf', gamma=0.8, C=C),
                        svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
            models = (clf.fit(X_train, y_train) for clf in models)
            prediction = []
            q=0
            svm_obj = None
            for clf in models:
                    svm_obj =clf
                    #BX[0]=min_max_scaler.inverse_transform(BX[0])
                    #BX[2]=min_max_scaler.inverse_transform(BX[2]) 
                    #BX[3]=min_max_scaler.inverse_transform(BX[3]) 
                    X0, X1,X2,X3 = BX[0][:, 0], BX[0][:, 1], BX[0][:, 2], BX[0][:, 3]
                    xx, yy,zz,ww,x_min, x_max, y_min, y_max ,z_min, z_max,w_min, w_max   = make_meshgrid(X0, X1,X2,X3)
                    try:
                        plot_contours(svm_obj,xx, yy,zz,ww,BX[2], BX[3], y_min, y_max ,z_min, z_max,w_min, w_max, type=0,eps=dat[2])
                        plot_contours(svm_obj,xx, yy,zz,ww,BX[2], BX[3],x_min, x_max, z_min, z_max,w_min, w_max, type=1,eps=dat[2])
                        plot_contours(svm_obj,xx, yy,zz,ww,BX[2], BX[3],x_min, x_max, y_min, y_max ,w_min, w_max, type=2,eps=dat[2])
                        plot_contours(svm_obj,xx, yy,zz,ww,BX[2], BX[3],x_min, x_max, y_min, y_max ,z_min, z_max, type=3,eps=dat[2])
                    except ValueError:
                        print("Oops!  That was no valid!!!.  Try again..." )
                        plt.clf()




    b = datetime.datetime.now()
    c = b - a
    print(c.seconds)
# =====================================================================
if __name__ == '__main__':
    # Download the data set from URL
    print("Start the process")
    plt.rcParams["mathtext.fontset"] = "cm"
    #With_Condition(eps_begin=0.0,eps_end=0.9,steps=100, quantity = 500, verify=False)
    eps = np.linspace(0,1, 100)
    Without_Condition(eps)
     
