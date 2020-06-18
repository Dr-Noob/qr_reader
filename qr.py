#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Paquetes estándar utilizados, y configuración de los gráficos:
import sys
import cv2
import math
import numpy as np
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def draw_str(dst, tuplexy, s):
    (x,y) = tuplexy
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)

def processing(inimg):
    _, img = cv2.threshold(inimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return cnts, img

def cnt_inside_polygon(cnt, polygon):  
  for pt in cnt:
      point = Point(pt[0][0], pt[0][1])  
      if not (polygon.contains(point)):
          return False
      
  return True

def pt_inside_img(pt, img):
    h, w = img.shape
    return (pt[0] > 0 and pt[0] < w) and \
           (pt[1] > 0 and pt[1] < h)

def find_qrcnts(imgtodraw, edge_candidates):
    # It does not matter which point do we take
    edg1 = edge_candidates[0][0]
    edg2 = edge_candidates[1][0]
    edg3 = edge_candidates[2][0]

    dist1 = math.hypot(edg1[0] - edg2[0], edg1[1] - edg2[1])    
    dist2 = math.hypot(edg1[0] - edg3[0], edg1[1] - edg3[1]) 
    dist3 = math.hypot(edg2[0] - edg3[0], edg2[1] - edg3[1]) 
    
    if dist1 > dist2 and dist1 > dist3:
        edge1 = edge_candidates[0]
        edge2 = edge_candidates[1]
        corner = edge_candidates[2]
    elif dist2 > dist1 and dist2 > dist3:
        edge1 = edge_candidates[0]
        edge2 = edge_candidates[2]
        corner = edge_candidates[1]
    else:
        edge1 = edge_candidates[1]
        edge2 = edge_candidates[2]
        corner = edge_candidates[0]
    
    max_ed1 = (0, 0)
    max_ed2 = (0, 0)
    max_dst = -1
    tmpi = -1
    tmpj = -1
    for i in range(0,4):
        for j in range(0,4):
            dst = math.hypot(edge1[i][0] - edge2[j][0], edge1[i][1] - edge2[j][1])
            if dst > max_dst:
                max_dst = dst
                max_ed1 = edge1[i]
                max_ed2 = edge2[j]
                tmpi = i
                tmpj = j

    # Detect the third one (corner)  
    d_arr = np.array([0, 0, 0, 0])
    d_arr[0] = math.hypot(max_ed1[0] - corner[0][0], max_ed1[1] - corner[0][1]) + \
               math.hypot(max_ed2[0] - corner[0][0], max_ed2[1] - corner[0][1]) 
    d_arr[1] = math.hypot(max_ed1[0] - corner[1][0], max_ed1[1] - corner[1][1]) + \
               math.hypot(max_ed2[0] - corner[1][0], max_ed2[1] - corner[1][1]) 
    d_arr[2] = math.hypot(max_ed1[0] - corner[2][0], max_ed1[1] - corner[2][1]) + \
               math.hypot(max_ed2[0] - corner[2][0], max_ed2[1] - corner[2][1]) 
    d_arr[3] = math.hypot(max_ed1[0] - corner[3][0], max_ed1[1] - corner[3][1]) + \
               math.hypot(max_ed2[0] - corner[3][0], max_ed2[1] - corner[3][1])             
    max_idx = np.argmax(d_arr)
    max_ed3 = corner[max_idx]                                          
    
    # Finally, find the last one 
    max_area = -1
    partial_cnt = np.array([max_ed1, max_ed3, max_ed2])
    pt1 = np.append(max_ed1, 1)          
    pt3 = np.append(max_ed2, 1)    
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            print('i=',i,'j=',j)
            pt2 = np.append(edge1[(tmpi+i)%4], 1)
            pt4 = np.append(edge2[(tmpj+j)%4], 1)
            
            l1 = np.cross(pt1, pt2)
            l2 = np.cross(pt3, pt4)
            
            ptfuga = np.cross(l1, l2) # float 
            print(ptfuga)
            
            if ptfuga[1] != 0:
              # Remove last coord
              if ptfuga[2] != 0:
                  ptfuga[0] = ptfuga[0]/ptfuga[2]
                  ptfuga[1] = ptfuga[1]/ptfuga[2] 
              ptfuga = ptfuga[:-1]
              
              # Check if its inside img!
              if(pt_inside_img(ptfuga, imgout)):
                  # If it is, keep the one with max area
                  tmp_ed4 = np.array((ptfuga[0], ptfuga[1])).astype('int32')                  
                  area = cv2.contourArea(np.append(partial_cnt, tmp_ed4).reshape(4,2))               
                  if area > max_area:
                      print('Found new with area ', area)
                      coord4x = ptfuga[0]
                      coord4y = ptfuga[1]  
                      max_area = area
                                                 
    max_ed4 = (coord4x, coord4y)
    outer_corners = [max_ed1, max_ed2, max_ed3, max_ed4]
    return outer_corners

def detect_qr(imgout, cnts):
    #print('Looking in ', len(cnts) , ' contours')
    if len(cnts) == 0:
        return

    edge_candidates = []
    for i, c in enumerate(cnts):              
      cnts_inside = 0    
      t_cnt = cnts[i].reshape(4,2)
      polygon = Polygon([tuple(t_cnt[0]), tuple(t_cnt[1]), tuple(t_cnt[2]), tuple(t_cnt[3])])
      for c in cnts:
          if(cnt_inside_polygon(c, polygon)):
              cnts_inside = cnts_inside+1
              
      # Check if current contour is a edge candidate
      if cnts_inside == 1 or cnts_inside == 2:
          edge_candidates.append(t_cnt)          
    
    #print('This polygon has ', len(edge_candidates), ' edges candidates')
      
    # We suppose that if we can find the three edges, we have a QR
    if len(edge_candidates) == 3:
        print('QR!')
        out = find_qrcnts(imgout, edge_candidates)
        if out is None:
            print('Failed to detect QR contours!')
            return None
        else:
            return np.array(out)
                        
    return None

def qr_wrap_perspective(img, H, qr_outside_cnt):
    s = 150
    h, w, c = img.shape    
    mymat = np.array([[s,   0,    0],
                      [0,  -s,  s],
                      [0,     0,    1]])
    
    perspect = np.zeros_like(img)
    cv2.warpPerspective(img, mymat @ np.linalg.inv(H), (w ,h), dst=perspect,  borderMode=cv2.BORDER_TRANSPARENT)
    return perspect[0:s,0:s]

def get_permutations(pts):
    perm = np.array(pts)
    rows, cols = pts.shape
    
    for i in range(0, rows-1):
        tmp = pts[i+1:rows]
        tmp = np.vstack([tmp, pts[0:i+1]])
        perm = np.append(perm, tmp)
    
    return perm.reshape(rows, rows, 2)

def qr_search_homography(imgout, qr_outside_cnt):
    pts = np.array([
               [0, 1],
               [1, 0],
               [0, 0],
               [1, 1],
              ])
    
    qr_outside_cnt = qr_outside_cnt.reshape(4,2)
    
    perms = get_permutations(qr_outside_cnt)
    harr = [0, 0, 0, 0]
    # Search for the best H
    for i, per in enumerate(perms):    
        tmpH, inliers = cv2.findHomography(pts, per, method=cv2.RANSAC,ransacReprojThreshold=5)        
        harr[i] = tmpH
    
    return harr[0]

# https://stackoverflow.com/questions/5228383/how-do-i-find-the-distance-between-two-points
def zoom(z, cnts, img):
    h, w, c = img.shape
    z = -z
    
    max_dst = -1
    min_dst = 100000
    for i,pt in enumerate(cnts):
        dst = math.hypot(pt[0], pt[1]) # distancia con punto 0,0
        if dst > max_dst:
            pt3 = i
            max_dst = dst
        if dst < min_dst:
            pt1 = i
            min_dst = dst
       
    max_dst = -1
    min_dst = 100000
    for i,pt in enumerate(cnts):
        dst = math.hypot(pt[0]-w, pt[1]) # distancia con punto 0,w
        if dst > max_dst:
            pt4 = i
            max_dst = dst
        if dst < min_dst:
            pt2 = i   
            min_dst = dst
            
    cnts[pt1] = cnts[pt1] - z
    cnts[pt2][0] = cnts[pt2][0] + z
    cnts[pt2][1] = cnts[pt2][1] - z
    cnts[pt3] = cnts[pt3] + z
    cnts[pt4][0] = cnts[pt4][0] - z
    cnts[pt4][1] = cnts[pt4][1] + z
    
def qr(imgout, cnts, z_zoom):
    origimg = imgout.copy()
    valid_cnts = []
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)            
            area = cv2.contourArea(c)
            if area > 300:
                ar = w / float(h)
                if (ar > .85 and ar < 1.3):
                    cv2.rectangle(imgout, (x, y), (x + w, y + h), (255,0,12), 1)            
                    valid_cnts.append(approx)
                    
    qr_outside_cnt = detect_qr(imgout, valid_cnts)                    
    
    if qr_outside_cnt is not None:
        zoom(-z_zoom, qr_outside_cnt,origimg)
        x,y,w,h = cv2.boundingRect(qr_outside_cnt)
        cv2.rectangle(imgout, (x, y), (x + w, y + h), (36,255,12), 3)                                      
        H = qr_search_homography(origimg, qr_outside_cnt)        
        #imgout = cv2.circle(imgout, tuple(qr_outside_cnt[0]), 5, (255, 0, 0), -1)
        #imgout = cv2.circle(imgout, tuple(qr_outside_cnt[1]), 10, (255, 0, 0), -1)
        #imgout = cv2.circle(imgout, tuple(qr_outside_cnt[2]), 15, (255, 0, 0), -1)
        #imgout = cv2.circle(imgout, tuple(qr_outside_cnt[3]), 20, (255, 0, 0), -1)
        return imgout, qr_wrap_perspective(origimg, H, qr_outside_cnt)
    return None

# Programa principal:
if __name__ == '__main__':

    # Creación de ventana y sliders asociados:
    def nothing(*arg):
        pass
    cv2.namedWindow('output') 
    cv2.namedWindow('qr') 
    cv2.moveWindow('output', 0, 0)
    cv2.moveWindow('qr', 0, 480)

    if len(sys.argv) > 1:
        source = int(sys.argv[1])
    else:
        source = 0
       
    cam = cv2.VideoCapture(source)             
    
    paused = False
    fig = None
    
    while True:
        if not paused:
            ret, frame = cam.read()
        if frame is None:
            print('End of video input')
            break
        
        z_zoom = 0
        imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cnts, imgout = processing(imgin)
        
        out = qr(frame, cnts, z_zoom)
                        
        if out is not None:
            frame, qrimg = out
            data = decode(qrimg, symbols=[ZBarSymbol.QRCODE])
            if not data:
                print('QR decode failed!')
            else:
                print('QR Found!')
                qrdata = data[0].data.decode()
                draw_str(frame, (20, 20), "QR Found!")
                draw_str(frame, (20, 40), "Text: {0}".format(str(qrdata)))                
            cv2.imshow('qr', qrimg)
            
        cv2.imshow('output', frame)           
        
        ch = cv2.waitKey(20) & 0xFF
        
        if ch == 27: 
            break
        elif ch == ord(' '):  
            paused = not paused
        elif ch == ord('.'):
            paused = True
            ret, frame = cam.read()

    cv2.destroyAllWindows()
    cam.release()
