#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from pyzbar.pyzbar import decode

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

def detect_qr(imgout, cnts):
    print('Looking in ', len(cnts) , ' contours')
    if len(cnts) == 0:
        return
    
    max_area = -1
    max_idx = -1
    
    for idx, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_idx = idx
           
    cnts_inside = 0
    t_cnt = cnts[max_idx].reshape(4,2)
    polygon = Polygon([tuple(t_cnt[0]), tuple(t_cnt[1]), tuple(t_cnt[2]), tuple(t_cnt[3])])
    for c in cnts:
      if(cnt_inside_polygon(c, polygon)):
          cnts_inside = cnts_inside+1
          
    print('This polygon has ', cnts_inside, ' polygons inside')
    
    if cnts_inside >= 2*3:
        #x,y,w,h = cv2.boundingRect(cnts[max_idx])
        #cv2.rectangle(imgout, (x, y), (x + w, y + h), (36,255,12), 3)                                      
        return cnts[max_idx]
    return None

def qr_wrap_perspective(img, H, qr_outside_cnt):
    h, w, c = img.shape
    mymat = np.array([[100,   0,    0],
                      [0,  -100,  100],
                      [0,     0,    1]])
    
    perspect = np.zeros_like(img)
    cv2.warpPerspective(img, mymat @ np.linalg.inv(H), (w ,h), dst=perspect,  borderMode=cv2.BORDER_TRANSPARENT)
    return perspect[0:100,0:100]

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
               [1, 0],
               [1, 1],
               [0, 1],
               [0, 0],
              ])
    qr_outside_cnt = qr_outside_cnt.reshape(4,2)
    
    perms = get_permutations(qr_outside_cnt)
    
    # Search for the best H
    max_len = 0
    usedi = 0
    for i, per in enumerate(perms):    
        tmpH, inliers = cv2.findHomography(pts, per, method=cv2.RANSAC,ransacReprojThreshold=5)
        counts = np.count_nonzero(inliers)
        if(counts > max_len):
            max_len = counts
            H = tmpH
            usedi = i

    print('Used i=', usedi)
    
    if max_len < 4:
        print('Failed to find H! (found ', max_len, ' inliers)')
        return None
    
    return H

def qr(imgout, cnts):
    origimg = imgout.copy()
    valid_cnts = []
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)            
            area = cv2.contourArea(c)
            if area > 1000:
                ar = w / float(h)
                if (ar > .85 and ar < 1.3):
                    cv2.rectangle(imgout, (x, y), (x + w, y + h), (255,0,12), 1)            
                    valid_cnts.append(approx)
                    
    qr_outside_cnt = detect_qr(imgout, valid_cnts)                    
    
    if qr_outside_cnt is not None:
        x,y,w,h = cv2.boundingRect(qr_outside_cnt)
        cv2.rectangle(imgout, (x, y), (x + w, y + h), (36,255,12), 3)                                      
        H = qr_search_homography(origimg, qr_outside_cnt)
        return imgout, qr_wrap_perspective(origimg, H, qr_outside_cnt)
        return imgout, origimg
    return None

if __name__ == '__main__':

    def nothing(*arg):
        pass
    cv2.namedWindow('output') 
    cv2.namedWindow('qr') 
    cv2.moveWindow("output", 0, 0)
    cv2.moveWindow("qr", 800, 0)
    
    if len(sys.argv) > 1:
        source = int(sys.argv[1])
    else:
        source = 0
       
    cam = cv2.VideoCapture(source)      
    
    paused = False
    fig = None
    flag = False
    
    while True:
        if not paused:
            ret, frame = cam.read()
        if frame is None:
            print('End of video input')
            break
        
        imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cnts, imgout = processing(imgin)
        
        out = qr(frame, cnts)
                        
        if out is not None:
            frame, qrimg = out
            data = decode(qrimg)
            if not data:
                print('QR decode failed!')
            else:
                print('QR Found!')
                flag = True
                qrdata = data[0].data.decode()
                lastdata = qrdata
                draw_str(frame, (20, 20), "QR Found!")
                draw_str(frame, (20, 40), "Text: {0}".format(str(qrdata)))                
            cv2.imshow('qr', qrimg)
        elif flag:
            qrdata = lastdata
            draw_str(frame, (20, 20), "QR Found!")
            draw_str(frame, (20, 40), "Text: {0}".format(str(qrdata)))
            
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
