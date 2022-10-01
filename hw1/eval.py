import random

if __name__ == '__main__':
    with open("prediction.csv", 'w') as f:
        f.write("name,label\n")
        for i in range(10000):
            f.write(f"{i + 1:05d}.mp4,{random.randint(0, 58)}\n")
