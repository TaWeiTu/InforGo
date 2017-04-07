import sys, json, numpy as np

def caculate(x):
    if x % 4 == 3:
        return 0
    return x+1

def main():
    print('0 0 0')
    # print(1)
    while 1:
        #get our data as an array from read_in()
        x, y, z = map(int, input().split())
        if x == -1 :
            break
        #output
        x = caculate(x)
        print(x + y * 4 + z * 16)

#start process
if __name__ == '__main__':
    main()