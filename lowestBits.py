# lowestBits.py
# author: Renata Paramastri
# Image steganography by encoding the secret in the least significant bits.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

SECRET = "figure_1.png"
MASK = "prague.png"
ENCODED = "encoded.png"
NUM_BITS = 2  # an integer between 0 and 8

def convertToUint8(floatArray):
    """ Converts an image array encoded in float32 to uint8. """
    scaled = floatArray * 255
    return scaled.astype('uint8')

def convertToFloat32(intArray):
    """ Converts an image array encoded in uint8 to float32. """
    scaled = intArray / 255
    return scaled.astype('float32')

def swapNumBits(secretArray, maskArray):
    """
    secretArray -- a 3D numpy array containing 8 bit strings.
    maskArray   -- a 3D numpy array containing 8 bit strings.
    maskArray must be the same size or bigger than secretArray,
    including its number of channels.

    Returns a bit array where the last NUM_BITS bits of each maskArray entry
    is changed to the first NUM_BITS of the corresponding entry in secretArray.
    """
    height, width, numChannels = secretArray.shape

    for row in range(height):
        for col in range(width):
            for channel in range(numChannels):
                secretBits = secretArray[row, col, channel][:NUM_BITS]
                firstMaskBits = maskArray[row, col, channel][:8-NUM_BITS]

                # encode secretBits in the last NUM_BITS bits of maskArray
                maskArray[row, col, channel] = firstMaskBits + secretBits

    return maskArray

def displayLowestBits(encodedArray):
    """
    encodedArray -- a 3D numpy array containing 8 bit strings.
    Returns another bit array where the first bits of each entry is taken
    from the lowest bits of the corresponding encodedArray pixel.
    """
    height, width, numChannels = encodedArray.shape
    numBitsArray = np.empty((height, width, numChannels), dtype='<U8')
    zeroString = '0' * (8 - NUM_BITS)  # for the rest of each 8 bit string

    for row in range(height):
        for col in range(width):
            for channel in range(numChannels):
                lowestBits = encodedArray[row,col,channel][-NUM_BITS:]
                numBitsArray[row,col,channel] = lowestBits + zeroString

    return numBitsArray

def encode(secret, mask):
    """
    secret -- string containing path to a png file.
    mask   -- string containing path to a png file.
    Returns a numpy array where secret is encoded in mask.
    Saves the encoded array as an image with filename ENCODED.
    """
    secretFloatArray = mpimg.imread(secret)
    maskFloatArray = mpimg.imread(mask)

    secretArray = convertToUint8(secretFloatArray)
    maskArray = convertToUint8(maskFloatArray)

    # function that will convert each value to an 8-bit string
    convertToBits = np.vectorize(lambda x: format(x, '08b'))
    secretBitArray = convertToBits(secretArray)
    maskBitArray = convertToBits(maskArray)

    encodedBitArray = swapNumBits(secretBitArray, maskBitArray)

    binToInt = np.vectorize(lambda x: int(x, 2))
    encodedIntArray = binToInt(encodedBitArray)

    encodedArray = convertToFloat32(encodedIntArray)

    mpimg.imsave(ENCODED, encodedArray)
    return encodedArray

def decode(encodedImage):
    """
    encodedImage -- string containing path to a png file.
    Returns a numpy array of the decode image.
    Saves the decoded image as 'decoded.png'
    """
    encodedFloatArray = mpimg.imread(encodedImage)
    encodedIntArray = convertToUint8(encodedFloatArray)

    # function that will convert each value to an 8-bit string
    convertToBits = np.vectorize(lambda x: format(x, '08b'))
    encodedBitArray = convertToBits(encodedIntArray)

    decodedBitArray = displayLowestBits(encodedBitArray)

    binToInt = np.vectorize(lambda x: int(x, 2))
    decodedIntArray = binToInt(decodedBitArray)

    decodedArray = convertToFloat32(decodedIntArray)
    mpimg.imsave('decoded.png', decodedArray)
    return decodedArray


def main():
    original = mpimg.imread(MASK)
    encodedArray = encode(SECRET, MASK)
    decodedArray = decode(ENCODED)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3)

    ax1.set_title('Original picture')
    ax2.set_title('Encoded (with {} bits)'.format(NUM_BITS))
    ax3.set_title('Decoded')

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    ax1.imshow(original)
    ax2.imshow(encodedArray)
    ax3.imshow(decodedArray)

    plt.show()

if __name__ == '__main__':
    main()
