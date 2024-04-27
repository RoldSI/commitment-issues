# Compiler settings
CC = gcc

# Target executable name
TARGET = real_test

# Source files
SRC = main.cpp

# Build target
all:
	$(CC) $(SRC) -o $(TARGET) 

# Clean up
clean:
	rm -f $(TARGET)
