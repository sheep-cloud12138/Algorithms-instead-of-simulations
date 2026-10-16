import sys, time, os, urllib.request, urllib.parse, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_review import run
B = {
 "6": [
   ("Q1","abs:\"physical units\" AND abs:\"neural operator\""),
   ("Q1","abs:\"scale equivariance\" AND abs:\"PDE\""),
   ("Q1","abs:\"unit-aware\" AND abs:\"neural network\""),
   ("Q1","abs:\"dimensionless numbers\" AND abs:\"deep learning\""),
   ("Q1","abs:\"invariance\" AND abs:\"units\" AND abs:\"machine learning\" AND abs:\"physics\""),
   ("Q1","ti:\"dimensional\" AND abs:\"operator learning\""),
   ("Q1","abs:\"similarity\" AND abs:\"dimensionless\" AND abs:\"neural network\""),
 ],
}
run("6", B["6"])
