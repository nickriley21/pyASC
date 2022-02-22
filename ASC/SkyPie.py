#! /usr/bin/env python3
#
# Takes about 15" for 1400 images on laptop with a local fast disk (100% cpu)
# But 60" on the Xeon, but at 300% cpu
#
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

#   plt.rcParams.update({'font.size': 10})
SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8



twopi = 2*np.pi

def plot1(table,ax1,ax2,Qtitle,title=None,invert=True,raw=False):
    # invert:      this will place dark sky on the outside of the pie
    
    #   table of decimal hour time and median sky brightness (50,000 is very bright)
    # (t,s,ffile) = np.loadtxt(table).T
    loaded = np.genfromtxt(table, dtype=None, delimiter=' ')
    try:
        (t,s,e,m) = np.array([t[0] for t in loaded]), np.array([t[1] for t in loaded]), np.array([t[2] for t in loaded]), np.array([t[3] for t in loaded])
        print(t)
        print("Time:",t.min(),t.max())
        print("Sky: ",s.min(),s.max())
        print("Exp: ",e.min(),e.max())
        print("Moon:",m.min(),m.max())
        amp = (m.min() + m.max())/2.0     # average moon phase
    except:
        # older format with only 2 columns
        (t,s) = np.array([t[0] for t in loaded]), np.array([t[1] for t in loaded])
        print(t)
        print("Time:",t.min(),t.max())
        print("Sky: ",s.min(),s.max())
        amp = -2.0
        
    t0 = t[0]
    t1 = t[-1]
    print(t0,t1)

    # tmin is the sunrise, from t1 (6), should be near 90
    # tmax is the sunset, from t0 (18)                270
    tmin = (6-t1)*15  +  90
    tmax = (18-t0)*15 + 270

    smax = 64000
    emax = e.max()

    print(tmin,tmax)
    x = (12-t) * twopi / 24.0
    if invert:
        #    dark sky on outside of the pie
        #y = s.max()-s
        y = smax-s
        print("y",invert,y.min(),y.max())

        p = e
        print("p",invert,p.min(),p.max())
    else:
        y = s
        print("y",invert,y.min(),y.max())

        p = e
        print("p",invert,p.min,p.max)

    
    print(x.min(),x.max())
    print(y.min(),y.max())



    ax1.plot(x, y)
    ax1.set_theta_zero_location('S')
    ax1.set_ylim([0,smax])
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.yaxis.set_major_formatter(plt.NullFormatter())
    ax1.yaxis.set_major_formatter(plt.NullFormatter())

    ax2.plot(x, p)
    ax2.set_theta_zero_location('S')
    ax2.set_ylim([0,emax])
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    
    if False:
        # always same pie, an extra hour either side
        tmin=75
        tmax=285
    print(tmin,tmax)
    ax1.set_thetamin(tmin)
    ax1.set_thetamax(tmax)

    ax2.set_thetamin(tmin)
    ax2.set_thetamax(tmax)

    y1 = y
    p1 = p
    if False:
        y1 = y*0 + smax
    
    ya = 0.2 * y1    
    yb = 0.4 * y1   
    yc = 0.8 * y1  
    yd = 0.8 * y1   
    ye = 0.9 * y1

    ax1.fill_between(x,0, ya,facecolor='green',alpha=0.1)
    ax1.fill_between(x,ya,yb,facecolor='green',alpha=0.3)
    ax1.fill_between(x,yb,yc,facecolor='green',alpha=0.5)
    ax1.fill_between(x,yc,yd,facecolor='green',alpha=0.7)
    ax1.fill_between(x,yd,ye,facecolor='green',alpha=0.85)
    ax1.fill_between(x,ye,y ,facecolor='green',alpha=1)

    pa = 0.2 * p1    
    pb = 0.4 * p1   
    pc = 0.8 * p1  
    pd = 0.8 * p1   
    pe = 0.9 * p1
    ax2.fill_between(x,0, pa,facecolor='orange',alpha=0.1)
    ax2.fill_between(x,pa,pb,facecolor='orange',alpha=0.3)
    ax2.fill_between(x,pb,pc,facecolor='orange',alpha=0.5)
    ax2.fill_between(x,pc,pd,facecolor='orange',alpha=0.7)
    ax2.fill_between(x,pd,pe,facecolor='orange',alpha=0.85)
    ax2.fill_between(x,pe,p ,facecolor='orange',alpha=1)
    if title != None and not raw:
        ax1.text(0,smax/2,title,horizontalalignment='center')
        #ax.set_title(title)

    if Qtitle and not raw:
        plt.suptitle("%s\nLocal Time: %.3f-%.3f h" % (table,t0,t1))
        ax1.set_title("Brightness: %g-%g" % (s.min(),s.max()),fontdict={'fontsize':8})
        ax2.set_title("Exposure: %g-%g" % (e.min(),e.max()),fontdict={'fontsize':8})
        
        # gets the number of the image to use. Will be a number between -15 and 15
        # calculated by multiplying the resulting moon illumination (negative or
        # positive depending on waning/waxing) by 15 and rounding to the nearest integer.
        # -15 and 15 are full moon, 0 is new moon.
        image_num = round(amp/(1/15))

        # if the image_num is out of bounds, just print the error moon value,
        # which we can use to debug.
        if True or image_num < -15 or image_num > 15:
            ax1.text(1.1, 0, 'moon %.3g' % amp, horizontalalignment='center', transform=ax1.transAxes)
        # else put the correct image
        else:
            # file names in moonphases directory are of the from moonphases<-15 to 15>.png
            moonphase_img = mpimg.imread('moonphases/moonphases' + str(int(image_num)) + '.png')

            if moonphase_img is not None:
                # image exists, put it on the plot
                imagebox = OffsetImage(moonphase_img, zoom = 0.35)
                ab = AnnotationBbox(imagebox, (0.5,0.5), xybox = (0,smax/4), frameon = False)
                ax.add_artist(ab)
            else:
                # image does not exist
                plt.text(0, smax/4, 'moon %.3g' % -30, horizontalalignment='center')

        # needs placement tweaking
        print('theta',tmin*3.14/180,tmax*3.14/180)
        deg_to_rad = 3.14/180
        ax1.text(3.14,              1.1*smax,   'midnight',        horizontalalignment='center',   fontdict={'fontsize':8})
        ax1.text(tmin*deg_to_rad,   smax,       'sunrise',         horizontalalignment='left',     fontdict={'fontsize':8})
        ax1.text(tmax*deg_to_rad,   smax,       'sunset',          horizontalalignment='right',    fontdict={'fontsize':8})

        ax2.text(3.14,              1.1*emax,   'midnight',        horizontalalignment='center',   fontdict={'fontsize':8})
        ax2.text(tmin*deg_to_rad,   emax,       'sunrise',         horizontalalignment='left',     fontdict={'fontsize':8})
        ax2.text(tmax*deg_to_rad,   emax,       'sunset',          horizontalalignment='right',    fontdict={'fontsize':8})
        


ntable = len(sys.argv[1:])
table = sys.argv[1]
png   = table + '.png'

if ntable == 1:
    Qtitle = True
else:
    Qtitle = False


Qraw = False    # set to true if you don't want any labeling around the plot, just the pie


if ntable > 1:
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


nx = int(np.sqrt(ntable))
ny = ntable // nx

print(nx,ny)

fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw=dict(projection='polar'))

if ntable > 1:
    plt.subplots_adjust(hspace = .001,wspace=0.001, left=0.01, right=0.99, bottom=0.01, top=0.99)
#      left  = 0.125  # the left side of the subplots of the figure
#      right = 0.9    # the right side of the subplots of the figure
#      bottom = 0.1   # the bottom of the subplots of the figure
#      top = 0.9      # the top of the subplots of the figure
#      wspace = 0.2   # the amount of width reserved for blank space between subplots
#      hspace = 0.2   # the amount of height reserved for white space between subplots

if Qtitle:
    plot1(table,ax1,ax2,True,raw=Qraw)
else:    
    k = 1
    for i in range(nx):
        for j in range(ny):
            plot1(sys.argv[k],ax1[j][i],ax2[j][i],False,sys.argv[k])
            k = k+1


plt.savefig(png)
# plt.show()

print("Written ",png)

# convert input.png -crop 400x400+128+64 -resize 40x40^   input.thumb.png