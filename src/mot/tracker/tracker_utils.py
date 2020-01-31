import math

def area(box):
    return math.sqrt((box[2]-box[0])*(box[3]-box[1]))

def ratio(box):
    return (box[2]-box[0])/(box[3]-box[1])

def center(box):
    return ((box[2]+box[0])/2, (box[3]+box[1])/2)

def center_dist(box1, box2):
    c1 = center(box1)
    c2 = center(box2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
