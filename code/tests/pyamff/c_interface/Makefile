#!/bin/make -f

# LIBS += libAMFF.a
ROOT_DIR=../..
BIN_DIR=$(ROOT_DIR)/bin
INC_DIR=$(ROOT_DIR)/include
LIB_DIR=$(ROOT_DIR)/lib
CC=g++
NAME=PyAMFF

SRC=$(wildcard *.cpp)
HEADS=$(wildcard *.h)
OBJ=$(SRC:.cpp=.o)
LIB := lib${NAME}.a


.PHONY: all headers 

all: headers $(LIB)

headers:
	cp $(HEADS) $(INC_DIR)

$(LIB) : $(OBJ)
	ar cru $@ $^
	ranlib $@
	cp $@ $(LIB_DIR)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean :
	rm -f $(OBJ)
	rm -f $(LIB) $(LIB_DIR)/$(LIB)
	cd $(INC_DIR) && rm -f $(HEADS)
	find ML-PYAMFF -type l -exec rm -f {} \;

