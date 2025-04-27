import copy
import logging
import math
import random
import time

from .fedbuff import FedBuffClient, FedBuffServer
from .fedgcr import FedGCRClient

from .base import BaseServer, BaseClient
import torch

Client = FedGCRClient
Server = FedBuffServer