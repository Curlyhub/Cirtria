
import os
import matplotlib.pyplot as plt
import numpy as np
import persim
import ripser
from persim import plot_diagrams


__author__ = "Yovani Torres Favier"
__copyright__ = "Copyright (C) 2020 CURLYHUB"
__license__ = "GPL v3"


FILEROOT = r"D:\Yovani\PhD\Master\Lectures\SciProb\Delivery\LongIV\Img_13"



def __min_birth_max_death(persistence):
    """This function returns (min_birth, max_death) from the persistence.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :returns: (float, float) -- (min_birth, max_death).
    """
    # Look for minimum birth date and maximum death date for plot optimisation
    max_death = 0
    min_birth = persistence[0][1][0]
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            if float(interval[1][1]) > max_death:
                max_death = float(interval[1][1])
        if float(interval[1][0]) > max_death:
            max_death = float(interval[1][0])
        if float(interval[1][0]) < min_birth:
            min_birth = float(interval[1][0])
    return (min_birth, max_death)

"""
Only 13 colors for the palette
"""
palette = ['#ff002b', '#1e7339', '#0000ff', '#1f00e6', '#ff00ff', '#ffff00',
           '#000000', '#880000', '#008800', '#000088', '#888800', '#880088',
           '#008888']

def interception(b,d):
    x = (b+d)*0.5
    y = min(d-x,x-b)
    return x,y

def media(b,d):
    x = (b+d)*0.5
    return x

def projector(a,b):
    return 2*b-a,0

def antiprojector(a,b):
    return b+a,0


 
def show_palette_values(alpha=0.6):
    """This function shows palette color values in function of the dimension.

    :param alpha: alpha value in [0.0, 1.0] for horizontal bars (default is 0.6).
    :type alpha: float.
    :returns: plot -- An horizontal bar plot of dimensions color.
    """
    colors = []
    for color in palette:
        colors.append(color)

    y_pos = np.arange(len(palette))

    plt.barh(y_pos, y_pos + 1, align='center', alpha=alpha, color=colors)
    plt.ylabel('Dimension')
    plt.title('Dimension palette values')

    plt.show()

def plot_persistence_barcode(persistence, alpha=0.6, height=0.5, title='Persistence barcode',filename="pre02.pdf"):
    """This function plots the persistence bar code.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for horizontal bars (default is 0.6).
    :type alpha: float.
    :returns: plot -- An horizontal bar plot of persistence.
    """
    name_Image = os.path.join(FILEROOT,filename)
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 1
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for bar code to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta
    # Draw horizontal bars in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            # Finite death case
            plt.barh(ind, (interval[1][1] - interval[1][0]), height=height,
                     left = interval[1][0], alpha=alpha,
                     color = palette[interval[0]])
        else:
            # Infinite death case for diagram to be nicer
            plt.barh(ind, (infinity - interval[1][0]), height=0.5,
                     left = interval[1][0], alpha=alpha,
                     color = palette[interval[0]])
        ind = ind + 1

    plt.title(title)
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, 0, ind])
    ax.yaxis.set_visible(False)
    plt.grid()
    plt.savefig(name_Image)
    plt.show()




def plot_persistence_barcodeModify(persistence, alpha=0.6, height=0.5, title='Persistence barcode',filename="pre01.pdf"):
    """This function plots the persistence bar code.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for horizontal bars (default is 0.6).
    :type alpha: float.
    :returns: plot -- An horizontal bar plot of persistence.
    """
    name_Image = os.path.join(FILEROOT,filename)
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for bar code to be more readable
    infinity = max_death + delta
    axis_start = min_birth - delta
    vlen =len(persistence)
    pat = '#ff002b'
    pal_color = []
    pal_color.append(pat)
    Homtxt = [r'$ H_0$',r'$ H_1$']
    q=0
    # Draw horizontal bars in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            # Finite death case
            ax.annotate('', xy=(interval[1][0],ind), xytext=(interval[1][1],ind),
            arrowprops={'arrowstyle': '-','color': palette[interval[0]]}, va='center')
        else:
            # Infinite death case for diagram to be nicer
            ax.annotate('', xy=(infinity,ind), xytext=(interval[1][0],ind),
            arrowprops={'arrowstyle': '-|>', 'lw': 2,'color': palette[interval[0]]}, va='center')
            plt.text(infinity, ind+0.5, r'$\infty$', color='k', alpha=alpha)
        if pat !=palette[interval[0]]:
            if palette[interval[0]] in pal_color:
                ind = ind + 1
            else:
                ind = ind + 3
                pal_color.append(palette[interval[0]])
                pat = palette[interval[0]]
                #plt.text(infinity/2, ind-10, Homtxt[q], color='k', alpha=alpha)
                if q <3:
                    q = q+1
        else:
            ind = ind + 1
    ax.annotate('', xy=(0,0), xytext=(0,ind+2),arrowprops={'arrowstyle': '-','lw': 0.5}, va='center')
    plt.title(title)
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, 0, ind])
    ax.yaxis.set_visible(False)
    plt.savefig(name_Image)
    plt.clf()

def plot_persistence_diagram(persistence, alpha=0.6):
    """This function plots the persistence diagram.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for points and horizontal infinity line (default is 0.6).
    :type alpha: float.
    :returns: plot -- An diagram plot of persistence.
    """
    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    # line display of equation : birth = death
    x = np.linspace(axis_start, infinity, 1000)
    # infinity line and text
    plt.plot(x, x, color='k', linewidth=1.0)
    plt.plot(x, [infinity] * len(x), linewidth=1.0, color='k', alpha=alpha)
    plt.text(axis_start, infinity, r'$\infty$', color='k', alpha=alpha)

    # Draw points in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            # Finite death case
            plt.scatter(interval[1][0], interval[1][1], alpha=alpha,
                        color = palette[interval[0]])
        else:
            # Infinite death case for diagram to be nicer
            plt.scatter(interval[1][0], infinity, alpha=alpha,
                        color = palette[interval[0]])
        ind = ind + 1

    plt.title('Persistence diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, axis_start, infinity + delta])
    return(plt)
#    plt.show()

# persistence_diagram and bootstrap
def plot_persistence_diagram_boot(persistence, alpha=0.6,band_boot=0,filename="pre03.pdf"):
    """This function plots the persistence diagram with confidence band 

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for points and horizontal infinity line (default is 0.6).
    :param band_boot: bootstrap band 
    :type alpha: float.
    :returns: plot -- An diagram plot of persistence.
    """
    name_Image = os.path.join(FILEROOT,filename)
    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    # line display of equation : birth = death
    x = np.linspace(axis_start, infinity, 1000)
    # infinity line and text
    plt.plot(x, x, color='k', linewidth=1.0)
    plt.plot(x, [infinity] * len(x), linewidth=1.0, color='k', alpha=alpha)
    # bootstrap band 
    plt.fill_between(x, x, x+band_boot, alpha = 0.3, facecolor='red')
    #,alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99',linewidth=0)   

    plt.text(axis_start, infinity, r'$\infty$', color='k', alpha=alpha)

    # Draw points in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            # Finite death case
            plt.scatter(interval[1][0], interval[1][1], alpha=alpha,
                        color = palette[interval[0]])
        else:
            # Infinite death case for diagram to be nicer
            plt.scatter(interval[1][0], infinity, alpha=alpha,
                        color = palette[interval[0]])
        ind = ind + 1

    plt.title('Persistence diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, axis_start, infinity + delta])
    plt.savefig(name_Image)
    plt.show()



#Warning this is not the most clever and efficient way to implement approximate landscapes.
def landscapes_approx(diag_dim,x_min,x_max,nb_steps,nb_landscapes):
    landscape = np.zeros((nb_landscapes,nb_steps))
    step = (x_max - x_min) / nb_steps
    #Warning: naive and not the best way to proceed!!!!!
    for i in range(nb_steps):
        x = x_min + i * step
        event_list = []
        for pair in diag_dim:
            b = pair[0]
            d = pair[1]
            if (b <= x) and (x<= d):
                if x >= (d+b)/2. :
                    event_list.append((d-x))
                else:
                    event_list.append((x-b))
        event_list.sort(reverse=True)
        event_list = np.asarray(event_list)
        for j in range(nb_landscapes):
            if(j<len(event_list)):
                landscape[j,i]=event_list[j]
    return landscape



#Ploting landscapes
def plot_persistence_landscapes(landscape, alpha=0.6,dgm1=[],filename="O5.pdf"):
    """This function plots the persistence diagram.

    :param persistence: The landscape to plot.
    :type alpha: float.
    :returns: plot -- A landscape plot of persistence.
    """
    name_Image = os.path.join(FILEROOT,filename)
    name_Interval = os.path.join(FILEROOT,"landscapefunction.txt")
    nbld = 4 # number of Landscapes 
    resolution = 10000
    length_max = 10
    ld_dim = 1 # landscape dim
    #plt.plot(np.linspace(dgm1[1].min(),dgm1[1].max() * 1, num=resolution),landscape[0:nbld,:].transpose(),linewidth=1.4, linestyle='dashed')
    plt.plot(np.linspace(dgm1[1].min(),dgm1[1].max() * 1, num=resolution),landscape[0:nbld,:].transpose(),linewidth=1.4, linestyle='dashed')
    A =[[0.92449987,1.0811106 ],
        [0.88572007,1.25574684],
        [0.75,0.92347169],
        [0.73518705,0.74953318],
        [0.71175838,0.73979729],
        [0.5978294,1.00642931],
        [0.47549975,0.4771792 ],
        [0.38327536,0.46872166],
        [0.36837479,0.44911024]]
    B=[]
    am = dgm1[1].min()
    B.append([dgm1[1].min(),0])
    for ds in dgm1[1][::-1]:
        B.append([ds[0],ds[1]])
    #B.append([dgm1[1].max(),0])
    k=0
    Ap = []
    Bp = []
    for v1 in B:
        if k== 0 or k==len(B):
            plt.plot(v1[0],v1[1], 'o', color='black');
            Ap.append([v1[0],v1[1]])
        else:
            plt.plot((v1[0]+v1[1])*0.5,(v1[1]-v1[0])*0.5, 'o', color='blue');
            Ap.append([(v1[0]+v1[1])*0.5,(v1[1]-v1[0])*0.5])
            Bp.append(v1[0])
            Bp.append(v1[1])
        k=k+1

    Bp.sort()
    px0,py0 = projector(Ap[0][0],Ap[1][0])
   

    x0,y0 = interception(B[2][0],B[1][1])
    x1,y1 = interception(B[6][0],B[5][1])
    x2,y2 = interception(B[8][0],B[4][1])
    x3,y3 = interception(B[8][0],B[7][1])
    x4,y4 = interception(B[9][0],B[4][1])

    px1,py1 = projector(px0,x0)
    px2,py2 = projector(px1,Ap[2][0])

    #write the interval of the landscape function
    file2 = open(name_Interval,"w+") 
    intervals = []
    #intervals.append([B[0][0],media()
    tx0,ty0 = Bp[0],Ap[1][0]
    tx1,ty1 = Ap[1][0],x0
    tx2,ty2 = x0,Ap[2][0]
    tx3,ty3 = Ap[2][0],Bp[4]
    tx4,ty4 = Bp[5],Ap[3][0]
    tx5,ty5 = Ap[3][0],Bp[6]
    tx6,ty6 = Bp[7],Ap[4][0]
    tx7,ty7 = Ap[4][0],x2
    tx8,ty8 = x2,Ap[8][0]
    tx9,ty9 = Ap[8][0],dgm1[1].max()

    intervals.append(str(B[0][0]) + " ("+ str(tx0)+"," + str(ty0) + ")\n")
    intervals.append(str(B[1][1]) + " ("+ str(tx1)+"," + str(ty1) + ")\n")
    intervals.append(str(B[2][0]) + " ("+ str(tx2)+"," + str(ty2) + ")\n")
    intervals.append(str(B[2][1]) + " ("+ str(tx3)+"," + str(ty3) + ")\n")
    intervals.append(str(B[3][0]) + " ("+ str(tx4)+"," + str(ty4) + ")\n")
    intervals.append(str(B[3][1]) + " ("+ str(tx5)+"," + str(ty5) + ")\n")
    intervals.append(str(B[4][0]) + " ("+ str(tx6)+"," + str(ty6) + ")\n")
    intervals.append(str(B[4][1]) + " ("+ str(tx7)+"," + str(ty7) + ")\n")
    intervals.append(str(B[8][0]) + " ("+ str(tx8)+"," + str(ty8) + ")\n")
    intervals.append(str(B[8][1]) + " ("+ str(tx9)+"," + str(ty9) + ")\n")
    dx0,dy0 = Bp[1],x0
    dx1,dy1 = x0,Bp[2]
    dx2,dy2 = Bp[7],Ap[5][0]
    dx3,dy3 = Ap[5][0],x1
    dx4,dy4 = x1,Ap[6][0]
    dx5,dy5 = Ap[6][0],Bp[8]
    dx6,dy6 = Bp[8],Ap[7][0]
    dx7,dy7 = Ap[7][0],x2
    dx8,dy8 = x2,x3
    dx9,dy9 = x3,x4
    dx10,dy10 = x4,Ap[9][0]
    dx11,dy11 = Ap[9][0],Bp[16]
    intervals.append(" -----------------------------------------------)\n")
    intervals.append(str(B[1][0]) + " ("+ str(dx0)+"," + str(dy0) + ")\n")
    intervals.append(str(B[1][1]) + " ("+ str(dx1)+"," + str(dy1) + ")\n")
    intervals.append(str(B[5][0]) + " ("+ str(dx2)+"," + str(dy2) + ")\n")
    intervals.append(str(B[5][1]) + " ("+ str(dx3)+"," + str(dy3) + ")\n")
    intervals.append(str(B[6][0]) + " ("+ str(dx4)+"," + str(dy4) + ")\n")
    intervals.append(str(B[6][1]) + " ("+ str(dx5)+"," + str(dy5) + ")\n")
    intervals.append(str(B[7][0]) + " ("+ str(dx6)+"," + str(dy6) + ")\n")
    intervals.append(str(B[7][1]) + " ("+ str(dx7)+"," + str(dy7) + ")\n")
    intervals.append(str(B[8][0]) + " ("+ str(dx8)+"," + str(dy8) + ")\n")
    intervals.append(str(B[4][1]) + " ("+ str(dx9)+"," + str(dy9) + ")\n")
    intervals.append(str(B[9][0]) + " ("+ str(dx10)+"," + str(dy10) + ")\n")
    intervals.append(str(B[9][1]) + " ("+ str(dx11)+"," + str(dy11) + ")\n")
    gx0,gy0 = Bp[8],x1
    gx1,gy1 = x1,Bp[9]
    gx2,gy2 = Bp[12],x3
    gx3,gy3 = x3,Bp[13]
    gx4,gy4 = Bp[13],x4
    gx5,gy5 = x4,Bp[15]
    intervals.append(" -----------------------------------------------)\n")

    intervals.append(str(B[1][0]) + " ("+ str(gx0)+"," + str(gy0) + ")\n")
    intervals.append(str(B[1][1]) + " ("+ str(gx1)+"," + str(gy1) + ")\n")
    intervals.append(str(B[5][0]) + " ("+ str(gx2)+"," + str(gy2) + ")\n")
    intervals.append(str(B[5][1]) + " ("+ str(gx3)+"," + str(gy3) + ")\n")
    intervals.append(str(B[6][0]) + " ("+ str(gx4)+"," + str(gy4) + ")\n")
    intervals.append(str(B[6][1]) + " ("+ str(gx5)+"," + str(gy5) + ")\n")
    file2.writelines(intervals) 
    file2.close() #to change file access modes 
    plt.grid()
    plt.savefig(name_Image)
    plt.show()
    #plt.title('Persistence landscape')
    
def plot_persistence_diagram_ripser(dgm1, alpha=0.6,filename="Persi.pdf"):
    """This function plots the persistence diagram.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for points and horizontal infinity line (default is 0.6).
    :type alpha: float.
    :returns: plot -- An diagram plot of persistence.
    """
    name_Image = os.path.join(FILEROOT,filename)
    Rip = ripser.Rips(maxdim=2,verbose=False)
    Rip.plot(dgm1)
    plt.savefig(name_Image)
    plt.show()


def plot_persistence_diagram_ripserUnique(dgm1, alpha=0.6,filename="Persi.pdf"):
    """This function plots the persistence diagram.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for points and horizontal infinity line (default is 0.6).
    :type alpha: float.
    :returns: plot -- An diagram plot of persistence.
    """
    name_Image = os.path.join(FILEROOT,filename)
    Rip = ripser.Rips(maxdim=2,verbose=False)
    Rip.plot(dgm1[1], labels=r'$H_1$')
    plt.savefig(name_Image)
    plt.show()
    


def truncated_simplex_tree(st,int_trunc=100):
    """This function return a truncated simplex tree  
    :st : a simplex tree
    :int_trunc : number of persistent interval keept per dimension (the largest) 
    """
    st.persistence()    
    dim = st.dimension()
    st_trunc_pers = [];
    for d in range(dim):
        pers_d = st.persistence_intervals_in_dimension(d)
        d_l= len(pers_d)
        if d_l > int_trunc:
            pers_d_trunc = [pers_d[i] for i in range(d_l-int_trunc,d_l)]
        else:
            pers_d_trunc = pers_d
        st_trunc_pers = st_trunc_pers + [(d,(l[0],l[1])) for l in pers_d_trunc]
    return(st_trunc_pers)


