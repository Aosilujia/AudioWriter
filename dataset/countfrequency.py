import os
import csv
import string

wordlist=string.ascii_letters
wordfrequency=[0]*len(wordlist)
total=0
word_count=0


file=open("..\CIR_writer\dataset\word1zq.csv","r")
#file=open("..\CIR_writer\dataset\wordlist.csv","r")
file_data=csv.reader(file)
for row in file_data:
    for word in row:
        if (word==""):
            continue;
        total+=len(word)
        word_count+=1
        for char in word:
            if char in wordlist:
                wordfrequency[wordlist.index(char)]+=1
            else:
                wordlist+=char
                wordfrequency.append(1)

for (index,value) in enumerate(wordlist):
    print(value,':',wordfrequency[index]/total)
print("average length:",total/word_count,"for",word_count,"words")
