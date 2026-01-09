#2025/12/27
import numpy as np
import cv2
TILE = 640
#funcs:
#_______________________________________________________________
def readImg(fileName):
    img = cv2.imread(fileName)
    if img is None:
        print("Error while reading image.")
        quit()
    else:
        return img

def getDim(img):
    """
    get img dimensions
    (obj) -> h, w
    """
    height, width, _ = img.shape
    return height, width

def getTiles(img,h,w):
    tiles = []
    for y in range(0,h,TILE):
        for x in range(0,w,TILE):
            tile = img[y:min(y+TILE, h),x:min(x+TILE, w)]
            #padding for partials
            th, tw = tile.shape[:2]
            padBottom = TILE-th
            padRight = TILE-tw
            tile = cv2.copyMakeBorder(tile,0,padBottom,0,padRight,cv2.BORDER_CONSTANT, value=(0,0,0))

            #this is the layout of tiles (tile,yoffset,xoffset)
            #this is the top left corner offset.
            tiles.append((tile,y,x))
    return tiles
#_______________________________________________________________

def main(imgFileName):
    """
    Takes as parameter imgFileName and returns array of tiles
    """
    fullImage = readImg(imgFileName)
    height, width = getDim(fullImage)

    #show dimensions and tiles in terminal
    print(height, width)

    #manage exception for exact dimensions
    tilesy = height//640+1
    tilesx = width//640+1
    if height % TILE == 0:
        tilesy = height//640
    if width % TILE == 0:
        tilesx = width//640
    
    expNumTiles = (tilesy,tilesx)
    totalTiles = expNumTiles[0]*expNumTiles[1]
    print(f"expected # of L1 tiles (y,x) -> {expNumTiles} ({totalTiles} tiles)")

    #get all tiles
    tiles = getTiles(fullImage,height,width)

    #this displays all the tiles on screen in seperate windows...
    """
    for i in range(totalTiles):
        print(tiles[i][1:3])
    
    cv2.imshow("full",fullImage)
    #cv2.imshow("Bottom Right",tiles[totalTiles-1][0])
    #cv2.imshow("Top Left",tiles[0][0])
    for i in range(totalTiles):
        cv2.imshow(f"tile{i}",tiles[i][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(tiles)
    print("showing the layout of tiles list")
    #pretty sure its like tiles[tile][1/2]
    print(tiles[14][1]) #this is yoffset
    print(tiles[14][2]) #this is xoffset
    """

    return tiles

if __name__ == "__main__":
    main("1.jpg")