import re
import csv
import string

pattern=r"[a-zA-Z]+"

file=open("..\CIR_writer\dataset\wordlist.csv","r")
wfile=open("../CIR_writer/dataset/temp.csv","w",newline='')
file_data=csv.reader(file)
csv_writer=csv.writer(wfile)
for row in file_data:
    for word in row:
        trueword=re.search(pattern,word)
        print([trueword[0]])
        csv_writer.writerow([trueword[0]])

wfile.close()
