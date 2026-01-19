mport uuid 
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import os
from werkzeug.utils import secure_filename
import random
from bson.objectid import ObjectId
import uuid
import json
import cv2
import supervision as sv
from roboflow import Roboflow
import numpy as np
from datetime import datetime
from flask import flash
from math import radians, sin, cos, sqrt, atan2
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import cloudinary
import cloudinary.uploader
import tempfile
