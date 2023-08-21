from flask import Flask, request
import lxml.etree
from lxml.etree import fromstring, XMLParser

app = Flask(__name__)

