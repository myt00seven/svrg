import os

NUM_EPOCHS = 20

def main():
    os.system("python mnist.py mlp sgd "+str(NUM_EPOCHS))
    os.system("python mnist.py mlpbn sgd "+str(NUM_EPOCHS))
    os.system("python draw.py")

main()