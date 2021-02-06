import os
import csv
import string

charlist=string.ascii_letters
wordlist=[]
wordfrequency=[0]*len(charlist)
total=0
word_count=0


file=open("..\CIR_writer\dataset\wordlistcollection50.csv","r")
#file=open("..\CIR_writer\dataset\wordlist.csv","r")
file2=open("..\CIR_writer\dataset\wordlist_nottrain.csv","r")

file_data=csv.reader(file)
file_data2=csv.reader(file2)
for row in file_data:
    for word in row:
        if (word==""):
            continue;
        total+=len(word)
        word_count+=1
        wordlist.append(word)
        for char in word:
            if char in charlist:
                wordfrequency[charlist.index(char)]+=1
            else:
                charlist+=char
                wordfrequency.append(1)

for row in file_data2:
    for word in row:
        if (word==""):
            continue;
        if (word in wordlist):
            print("duplicate:"+word)

for (index,value) in enumerate(charlist):
    print(value,':',wordfrequency[index]/total)
print("average length:",total/word_count,"for",word_count,"words")
