import numpy as np
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy

def perlin(x,y,seed=None):
    # permutation table
    if seed is not None:
        np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u) 
    return lerp(x1,x2,v)

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

def make_perlin(s,h,w):
    lin = np.linspace(0,s,max(h,w),endpoint=False)
    x,y = np.meshgrid(lin,lin)

    def norm(v):
        v -= v.min()
        v /= v.max()
        return v

    r = norm(perlin(x,y))
    g = norm(perlin(x,y))
    b = norm(perlin(x,y))

    img = np.stack([r, g, b], -1)

    return img[0:h,0:w]

if __name__ == "__main__":
    import cv2
    img_raw = (make_perlin(100, 1024, 2048), make_perlin(10, 1024, 2048))
    #img_raw = (make_perlin(10, 1024, 2048))
    # img_raw = cv2.resize(img_raw, (2048, 1024))

    m = np.mean(img_raw,(0,1,2))
    _channel_means = np.array([123.68, 116.779, 103.939])/255
    norm = np.clip(img_raw - m + _channel_means,0,1)
    # img_raw = norm

    # plt.figure()
    # plt.imshow(img_raw[0],origin='upper')
    # plt.figure()
    # plt.imshow(img_raw[1])
    # plt.show()
    cv2.imwrite("perlin_grey.png", (img_raw[1][...,0]*255).astype(np.uint8))
    cv2.imwrite("perlin_100.png", (img_raw[0]*255).astype(np.uint8))
    cv2.imwrite("perlin_10.png", (img_raw[1]*255).astype(np.uint8))