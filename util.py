import scipy

IMG_HEIGHT = 84
IMG_WIDTH = 84

def process_image(img):
    rgb = scipy.misc.imresize(img, [IMG_WIDTH, IMG_HEIGHT], interp='bilinear')

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    o = gray.astype('float32') / 255
    return o