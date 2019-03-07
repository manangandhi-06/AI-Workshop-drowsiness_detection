# -*- coding: utf-8 -*-

import playsound 
from scipy.spatial import distance as dist

def play_alarm(file_path):
    playsound.playsound(file_path)
    
def calculate_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    eye_aspect_ratio = (A+B)/(2.0*C)
    
    return eye_aspect_ratio