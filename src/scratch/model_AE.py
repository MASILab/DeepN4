import torch.nn as nn
import torch


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        in_channels = 64 * 64 * 64 
        out_channels = 128 #256 #1024 #64

        #Encoder
        self.flat = nn.Flatten()
        self.en1 = nn.Linear(in_channels, out_channels)
        self.norm = nn.InstanceNorm1d(out_channels)
        
        # Decoder
        self.de1 = nn.Linear(out_channels, in_channels)
        self.relu = nn.ReLU()
        return
        

    def encode(self, x):
        f = self.flat(x)
        e = self.en1(f)
        n = self.norm(e)
        encode_out = self.relu(n)
        return encode_out
    

    def decode(self, x):
        decode_out = self.relu(self.de1(x))
        return decode_out


    def forward(self, x):
        z = self.encode(x)
        r = self.decode(z)
        out = torch.reshape(r, (64, 64, 64))
        # bin = torch.round(torch.sigmoid(r)) # THIS IS A BAD IDEA
        return out
