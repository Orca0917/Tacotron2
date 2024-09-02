import matplotlib.pyplot as plt

def plot_specgram(mel_specgram, file_path="mel_specgram.png"):
    plt.imshow(mel_specgram.cpu().detach(), origin='lower', aspect='auto')
    plt.savefig(file_path) 
    plt.close()