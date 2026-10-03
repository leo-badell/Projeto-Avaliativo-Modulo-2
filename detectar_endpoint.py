from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display
