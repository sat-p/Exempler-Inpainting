FILES="test.cxx criminisi.cxx"
OUTPUT="INPAINT"

g++ -Wall -g -O3 -std=c++14 ${FILES} -o ${OUTPUT} `pkg-config opencv --cflags --libs`